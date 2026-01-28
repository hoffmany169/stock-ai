import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import Calendar
from datetime import date


import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import Calendar
from datetime import date


class DateRangePicker(ttk.Frame):
    """
    Reusable ttk control for selecting a start and end date.
    - End date defaults to start date on first selection
    - No auto-correction after that
    - Invalid ranges show warning popup
    """

    def __init__(self, master, *, start_text="Start Date", end_text="End Date", width=28):
        """
        Docstring for __init__
        
        :param master: parent window
        :param start_text: text shown on start button
        :param end_text: text shown on end button
        :param width: width of two buttons
        """
        super().__init__(master)

        self._start_date = None
        self._end_date = None

        self._start_text = start_text
        self._end_text = end_text
        self._width = width

        self._setup_style()
        self._create_widgets()

    # ---------- public API ----------

    def get(self):
        if not self._start_date or not self._end_date:
            return None, None
        return (
            self._start_date.strftime("%Y-%m-%d"),
            self._end_date.strftime("%Y-%m-%d"),
        )

    def get_dates(self):
        return self._start_date, self._end_date

    def set(self, start_date, end_date=None):
        self._start_date = start_date
        self._end_date = end_date or start_date
        self._update_buttons()

    # ---------- internal ----------

    def _setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("DateRange.TButton", padding=6)

    def _create_widgets(self):
        self.start_btn = ttk.Button(
            self,
            text=self._start_text,
            style="DateRange.TButton",
            width=self._width,
            command=lambda: self._open_calendar("start"),
        )
        self.start_btn.grid(row=0, column=0, padx=(0, 8))

        self.end_btn = ttk.Button(
            self,
            text=self._end_text,
            style="DateRange.TButton",
            width=self._width,
            command=lambda: self._open_calendar("end"),
        )
        self.end_btn.grid(row=0, column=1)

    def _open_calendar(self, target):
        popup = tk.Toplevel(self)
        popup.title("Select Date")
        popup.transient(self)
        popup.grab_set()
        popup.resizable(False, False)

        today = date.today()
        current = (
            self._start_date if target == "start" else self._end_date
        ) or today

        cal = Calendar(
            popup,
            selectmode="day",
            year=current.year,
            month=current.month,
            day=current.day,
        )
        cal.pack(padx=10, pady=10)

        def confirm():
            selected = cal.selection_get()

            # --- validation ---
            if target == "start" and self._end_date:
                if selected > self._end_date:
                    messagebox.showwarning(
                        "Invalid Date Range",
                        "Start date cannot be later than end date.",
                        parent=popup,
                    )
                    return

            if target == "end" and self._start_date:
                if selected < self._start_date:
                    messagebox.showwarning(
                        "Invalid Date Range",
                        "End date cannot be earlier than start date.",
                        parent=popup,
                    )
                    return

            # --- apply selection ---
            if target == "start":
                self._start_date = selected

                # ✅ default end date (only if not set yet)
                if self._end_date is None:
                    self._end_date = selected

            else:
                self._end_date = selected

            self._update_buttons()
            popup.destroy()

        ttk.Button(popup, text="Select", command=confirm).pack(pady=(0, 10))
        popup.bind("<Return>", lambda e: confirm())
        popup.bind("<Escape>", lambda e: popup.destroy())

    def _update_buttons(self):
        self.start_btn.config(
            text=f"{self._start_text}: {self._start_date:%Y-%m-%d}"
            if self._start_date
            else self._start_text
        )

        self.end_btn.config(
            text=f"{self._end_text}: {self._end_date:%Y-%m-%d}"
            if self._end_date
            else self._end_text
        )

if __name__ == "__main__":
    root = tk.Tk()
    root.title("DateRangePicker – Warning Mode")

    picker = DateRangePicker(root)
    picker.pack(padx=20, pady=20)

    ttk.Button(
        root,
        text="Read Value",
        command=lambda: print(picker.get(), picker.get_dates())
    ).pack()

    root.mainloop()
