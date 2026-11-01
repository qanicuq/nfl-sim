# nfl-sim

This project plays every NFL game 10,000 times in a computer. It checks whether those simulations would have made money on past games. It can also write picks for the coming week and send them to a phone.

It uses only the free public nflverse games file. No paid data, no account, and no key.

## How to run

Open the Actions tab.

1. Run Backtest first. That chooses the settings and writes the past-season results.
2. Run Predict next. That writes picks for games from today through the next 7 days. For a phone alert, add a repository secret named NTFY_TOPIC before you run Predict.

Both jobs run only when you start them by hand. Nothing runs on a timer.

This is not betting advice.