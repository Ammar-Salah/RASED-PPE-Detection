# Contributing to RASED AI — PPE Detection System

Thank you for your interest in contributing! Here's how you can help.

## 🐛 Reporting Bugs

1. Check existing [Issues](../../issues) to avoid duplicates.
2. Open a new issue with:
   - Steps to reproduce
   - Expected vs. actual behavior
   - OS, Python version, GPU info

## 💡 Feature Requests

Open an issue with the **enhancement** label describing:
- The problem you're trying to solve
- Your proposed solution
- Any alternatives considered

## 🔧 Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/RASED-PPE-Detection.git
cd RASED-PPE-Detection
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## 📝 Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📏 Code Style

- Follow PEP 8
- Add docstrings to all public functions
- Include type hints where possible
- Keep functions focused and small

## 🧪 Testing

- Test with webcam, video files, and RTSP streams
- Verify the dashboard renders correctly
- Check Telegram alerts if modifying the alert system
