# -*- coding: utf-8 -*-
"""
license_manager.py — RSA-лицензия Q-Net Lab. В файле ровно один класс:
:class:`LicenseManager`.

Как устроена лицензия (простая проверка подписи, без «хитростей»)
=================================================================
* У издателя (разработчика) есть пара RSA-ключей. **Закрытый** ключ хранится
  только у издателя и никогда не входит в поставку приложения. **Открытый**
  ключ ``public_key.pem`` лежит рядом с ``main.py``.
* Лицензия — JSON-файл ``license.json``::

      {
        "payload":   {"product": "Q-Net Lab", "subject": "Лаборатория …",
                      "license_id": "…", "issued_at": "…Z",
                      "not_before": "…Z", "expires_at": "…Z",
                      "format_version": "1"},
        "signature": "base64…"
      }

* Издатель подписывает закрытым ключом **канонический JSON** поля payload
  (ключи по алфавиту, без пробелов, UTF-8). Схема подписи фиксирована в коде:
  RSA-PSS, MGF1-SHA256, хэш SHA-256, соль 32 байта.
* При запуске приложение проверяет подпись открытым ключом. Любое изменение
  payload (имени, срока) делает подпись недействительной. Затем проверяются
  продукт и срок действия.

Чего эта схема НЕ делает (и это осознанно): не привязывается к компьютеру,
не обращается к сети, не защищает от правки исходников Python или перевода
системных часов. Это обычная офлайн-лицензия на основе подписи.

Команды издателя (через main.py, GUI не запускается)::

    python main.py --generate-keys --private-key ../issuer/private_key.pem
    python main.py --issue-license --private-key ../issuer/private_key.pem --subject "Lab" --days 365
    python main.py --verify-license
"""
from __future__ import annotations

import base64
import binascii
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


class LicenseManager:
    """Проверка RSA-подписи и срока лицензии; выпуск лицензий издателем.

    Пример::

        manager = LicenseManager()                      # ключ рядом с main.py
        result = manager.verify("license.json", progress=print)
        if result["valid"]:
            ...  # запускаем GUI

    Класс ничего не создаёт и не записывает при проверке: если лицензии нет,
    пробная лицензия или ключ автоматически НЕ появятся.
    """

    PRODUCT = "Q-Net Lab"
    FORMAT_VERSION = "1"
    #: Ограничение размера файлов: защита от случайного выбора огромного файла.
    MAX_FILE_BYTES = 64 * 1024
    #: Обязательные поля подписанной части лицензии.
    REQUIRED_FIELDS = ("product", "format_version", "subject", "license_id",
                       "issued_at", "not_before", "expires_at")

    def __init__(self, public_key_path: str | Path | None = None) -> None:
        """
        :param public_key_path: путь к открытому ключу издателя. По умолчанию —
            ``public_key.pem`` в папке этого файла (а не в текущем рабочем
            каталоге: PyCharm может запускать программу из другой папки).
        """
        base = Path(__file__).resolve().parent
        self.public_key_path = Path(public_key_path).resolve() if public_key_path else base / "public_key.pem"
        self.default_license_path = base / "license.json"

    # ------------------------------------------------------------------
    # Общие для подписи и проверки операции
    # ------------------------------------------------------------------
    @staticmethod
    def _padding() -> padding.PSS:
        """Фиксированная схема подписи: RSA-PSS с MGF1(SHA-256) и солью 32 байта."""
        return padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32)

    @staticmethod
    def canonical(payload: dict) -> bytes:
        """Байты, которые подписываются: JSON с сортировкой ключей без пробелов.

        Благодаря канонической форме лицензию можно переформатировать
        (отступы, переносы строк) — подпись от этого не ломается.
        """
        return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")

    @staticmethod
    def _parse_time(value) -> datetime:
        """ISO-8601 с явным часовым поясом (например, ``2026-01-01T00:00:00Z``) → UTC."""
        if not isinstance(value, str):
            raise ValueError("дата в лицензии должна быть строкой ISO-8601")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("дата в лицензии должна содержать часовой пояс (UTC)")
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _format_time(value: datetime) -> str:
        return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    # ------------------------------------------------------------------
    # Проверка (выполняется при каждом запуске приложения)
    # ------------------------------------------------------------------
    def verify(self, license_path: str | Path | None = None,
               progress: Callable[[int, str], None] | None = None,
               now: datetime | None = None) -> dict:
        """Проверить лицензию.

        :param license_path: путь к license.json; None — файл рядом с main.py.
        :param progress: функция ``progress(процент, текст)`` для полосы
            загрузки на заставке. Вызывается на каждом этапе проверки.
        :param now: момент проверки (только для тестов); по умолчанию — текущее
            время UTC.
        :returns: ``{"valid": bool, "message": str, "payload": dict | None,
            "days_left": int | None}``. Метод не бросает исключений: любая
            ошибка превращается в ``valid=False`` с понятным сообщением.
        """
        def step(percent: int, text: str) -> None:
            if progress is not None:
                progress(percent, text)

        path = Path(license_path) if license_path else self.default_license_path
        try:
            step(10, "Чтение открытого ключа издателя…")
            if not self.public_key_path.is_file():
                raise ValueError(f"не найден открытый ключ {self.public_key_path.name} рядом с main.py")
            key = serialization.load_pem_public_key(self.public_key_path.read_bytes())
            if not isinstance(key, rsa.RSAPublicKey) or key.key_size < 2048:
                raise ValueError("нужен открытый RSA-ключ длиной не менее 2048 бит")

            step(30, "Чтение файла лицензии…")
            if not path.is_file():
                raise ValueError(f"файл лицензии не найден: {path}")
            if path.stat().st_size > self.MAX_FILE_BYTES:
                raise ValueError("файл лицензии слишком большой")
            envelope = json.loads(path.read_text(encoding="utf-8-sig"))  # -sig: Блокнот может добавить BOM
            if not isinstance(envelope, dict) or set(envelope) != {"payload", "signature"}:
                raise ValueError("лицензия должна содержать ровно поля payload и signature")
            payload, signature = envelope["payload"], envelope["signature"]
            if not isinstance(payload, dict) or not isinstance(signature, str):
                raise ValueError("неверные типы полей payload/signature")

            step(50, "Проверка RSA-подписи…")
            key.verify(base64.b64decode(signature, validate=True), self.canonical(payload),
                       self._padding(), hashes.SHA256())

            step(70, "Проверка продукта и срока действия…")
            missing = [f for f in self.REQUIRED_FIELDS if f not in payload]
            if missing:
                raise ValueError(f"в лицензии нет полей: {', '.join(missing)}")
            if payload["product"] != self.PRODUCT or payload["format_version"] != self.FORMAT_VERSION:
                raise ValueError("лицензия выпущена для другого продукта или версии формата")
            starts = self._parse_time(payload["not_before"])
            expires = self._parse_time(payload["expires_at"])
            current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
            if current < starts:
                raise ValueError(f"срок действия ещё не начался ({payload['not_before']})")
            if current >= expires:
                raise ValueError(f"срок действия истёк {payload['expires_at']}")
            days_left = (expires - current).days
            step(80, "Лицензия действительна")
            return {"valid": True, "payload": payload, "days_left": days_left,
                    "message": f"Лицензия: {payload['subject']} (осталось дней: {days_left})"}
        except InvalidSignature:
            message = "Подпись лицензии недействительна: файл изменён или подписан другим ключом."
        except (OSError, ValueError, TypeError, binascii.Error, json.JSONDecodeError) as exc:
            message = f"Лицензия не прошла проверку: {exc}"
        except Exception as exc:  # noqa: BLE001 — любая неожиданная ошибка закрывает запуск
            message = f"Ошибка проверки лицензии: {type(exc).__name__}: {exc}"
        return {"valid": False, "payload": None, "days_left": None, "message": message}

    # ------------------------------------------------------------------
    # Операции издателя (не нужны конечному пользователю)
    # ------------------------------------------------------------------
    @staticmethod
    def generate_keys(private_key_path: str | Path, public_key_path: str | Path, key_size: int = 3072) -> None:
        """Создать пару RSA-ключей издателя. Существующие файлы НЕ перезаписываются.

        :param private_key_path: куда записать ЗАКРЫТЫЙ ключ (держите вне поставки!).
        :param public_key_path: куда записать открытый ключ (кладётся рядом с main.py).
        :raises FileExistsError: если любой из файлов уже существует.
        """
        private_path, public_path = Path(private_key_path).resolve(), Path(public_key_path).resolve()
        if private_path == public_path:
            raise ValueError("закрытый и открытый ключи должны быть разными файлами")
        if private_path.exists() or public_path.exists():
            raise FileExistsError("файл ключа уже существует — выберите другие имена")
        key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
        private_path.parent.mkdir(parents=True, exist_ok=True)
        public_path.parent.mkdir(parents=True, exist_ok=True)
        # O_EXCL: файл создаётся, только если его ещё нет (защита от перезаписи).
        descriptor = os.open(private_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
        with public_path.open("xb") as stream:
            stream.write(key.public_key().public_bytes(serialization.Encoding.PEM,
                                                       serialization.PublicFormat.SubjectPublicKeyInfo))

    @classmethod
    def issue(cls, private_key_path: str | Path, output_path: str | Path, subject: str,
              days: int = 365, overwrite: bool = False) -> dict:
        """Выпустить подписанную лицензию.

        :param private_key_path: закрытый ключ издателя (PEM, без пароля).
        :param output_path: куда записать license.json.
        :param subject: владелец лицензии (организация, лаборатория).
        :param days: срок действия в днях от текущего момента.
        :param overwrite: разрешить перезапись существующего файла.
        :returns: подписанный payload.
        """
        if not isinstance(subject, str) or not subject.strip():
            raise ValueError("укажите владельца лицензии (subject)")
        if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= 36500:
            raise ValueError("срок должен быть от 1 до 36500 дней")
        key = serialization.load_pem_private_key(Path(private_key_path).read_bytes(), password=None)
        if not isinstance(key, rsa.RSAPrivateKey):
            raise ValueError("нужен закрытый RSA-ключ")
        now = datetime.now(timezone.utc).replace(microsecond=0)
        payload = {"product": cls.PRODUCT, "format_version": cls.FORMAT_VERSION,
                   "subject": subject.strip(), "license_id": str(uuid.uuid4()),
                   "issued_at": cls._format_time(now), "not_before": cls._format_time(now),
                   "expires_at": cls._format_time(now + timedelta(days=days))}
        signature = key.sign(cls.canonical(payload), cls._padding(), hashes.SHA256())
        envelope = {"payload": payload, "signature": base64.b64encode(signature).decode("ascii")}
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w" if overwrite else "x", encoding="utf-8", newline="\n") as stream:
            json.dump(envelope, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        return payload
