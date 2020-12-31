mod event;
mod parse;
mod train;

use event::{Event, Events};
use std::{error::Error, io};
use termion::{event::Key, input::MouseTerminal, raw::IntoRawMode, screen::AlternateScreen};
use tui::{
    backend::{Backend, TermionBackend},
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    symbols,
    text::Span,
    widgets::{Axis, Block, Borders, Chart, Dataset},
    Frame, Terminal,
};

fn main() -> Result<(), Box<dyn Error>> {
    // Terminal initialization
    let stdout = io::stdout().into_raw_mode()?;
    let stdout = MouseTerminal::from(stdout);
    let stdout = AlternateScreen::from(stdout);
    let backend = TermionBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let events = Events::new();

    let mut app = App::new();

    loop {
        match events.next()? {
            Event::Input(input) => {
                if input == Key::Char('q') {
                    break;
                }
            }
            Event::Tick => {
                terminal.draw(|f| app.draw(f))?;
            }
            Event::EpochComplete(cost) => {
                app.add_cost(cost);
            }
        }
    }

    Ok(())
}

struct App {
    costs: Vec<(f64, f64)>,
}

impl App {
    fn new() -> Self {
        App { costs: vec![] }
    }
    fn add_cost(&mut self, cost: f64) {
        self.costs
            .push((self.costs.len() as f64 + 1.0, cost.log10()));
    }
    fn draw<B: Backend>(&self, f: &mut Frame<B>) {
        let size = f.size();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Ratio(1, 1)].as_ref())
            .split(size);

        let datasets = vec![Dataset::default()
            .marker(symbols::Marker::Dot)
            .style(Style::default().fg(Color::Cyan))
            .data(&self.costs)];

        let w = size.width / 2;
        let width = w as f64;
        let max_epoch = self.costs.last().map_or(0.0, |(epoch, _)| *epoch);
        let (start, end) = if max_epoch > width {
            (max_epoch - width, max_epoch)
        } else {
            (0.0, width)
        };

        let chart = Chart::new(datasets)
            .block(
                Block::default()
                    .title(Span::styled(
                        "Chart 1",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ))
                    .borders(Borders::ALL),
            )
            .x_axis(
                Axis::default()
                    .title("Epoch")
                    .style(Style::default().fg(Color::Gray))
                    .labels(vec![
                        Span::raw(format!("{}", start)),
                        Span::raw(format!("{}", end)),
                    ])
                    .bounds([start, end]),
            )
            .y_axis(
                Axis::default()
                    .title("Cost")
                    .style(Style::default().fg(Color::Gray))
                    .labels(vec![
                        Span::raw("1e-5"),
                        Span::raw("1e-4"),
                        Span::raw("0.01"),
                        Span::raw("0.1"),
                        Span::raw("1.0"),
                        Span::raw("10.0"),
                    ])
                    .bounds([-5.0, 1.0]),
            );
        f.render_widget(chart, chunks[0]);
    }
}
