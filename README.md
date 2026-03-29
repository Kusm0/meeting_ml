# Diploma: аналіз даних і ML-сервіс для зустрічей

У репозиторії — код дипломного проєкту: допоміжний скрипт у `data/` та основний сервіс **`meeting_ml_service`** (класифікація транскриптів зустрічей: рішення, тип теми, діалогові акти; TF-IDF + DistilBERT).

## Швидкий старт

```bash
cd meeting_ml_service
pip install -r requirements.txt
python -m src.web.gradio_app
```

Або через Docker:

```bash
cd meeting_ml_service
docker compose -f docker/docker-compose.yml up --build
```

(Якщо використовуєш класичний синтаксис: `docker-compose` замість `docker compose`.)

## Що не входить у Git

Датасети, навчені ваги моделей, метрики та артефакти тренувань залишаються локально — див. `.gitignore`.

## Репозиторій

<https://github.com/Kusm0/meeting_ml>
