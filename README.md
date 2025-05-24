# X-Automation

An advanced automation system for X (formerly Twitter) that includes crisis detection, analytics, scheduling, and strategy improvement capabilities.

## Features

- **Crisis Detection**
  - Sentiment analysis for comments and quote retweets
  - Red alert system for potential crises
  - 7-day analysis window for trend detection

- **Analytics & Metrics**
  - Performance metrics collection
  - Competitor metrics comparison
  - Engagement rate calculations

- **Calendar & Scheduling**
  - Japanese holiday awareness
  - Event day handling
  - Custom scheduling preferences

- **User Management**
  - Authentication system
  - User persona management
  - User settings storage

- **Advanced Features**
  - Trend inference system
  - Strategy improvement engine

## Setup

1. Clone the repository:
```bash
git clone https://github.com/MBilalKhanAI/X-Automation.git
cd X-Automation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
```

4. Run the application:
```bash
python x_optimized_v1.py
```

## Project Structure

- `x_optimized_v1.py`: Main application file
- `agents.py`: Agent implementations (Learning, Persona, Tweet)
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (not tracked by git)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License 