import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats as stats
from pandas.plotting import register_matplotlib_converters



pd.options.display.float_format = '{:,.2f}'.format

# Create locators for ticks on the time axis

register_matplotlib_converters()

df_yearly = pd.read_csv('annual_deaths_by_clinic.csv')
# parse_dates avoids DateTime conversion later
df_monthly = pd.read_csv('monthly_deaths.csv', parse_dates=['date'])

## identify the shape, column names, years, NAN values, duplicates, average births and deaths per month
print(df_yearly.shape)
print(df_monthly.shape)
print(df_yearly)
print(df_monthly)
print(df_yearly.duplicated().any())
print(df_monthly.duplicated().any())
print(df_yearly.isna().any())
print(df_monthly.isna().any())
print(df_monthly.describe())
print(df_yearly.describe())
print(df_monthly.info())
print(df_yearly.info())

## calculate percentage of women who died giving birth in the yearly dataset
death_prob = df_yearly["deaths"].sum()/df_yearly["births"].sum() * 100
print({death_prob:.3})

## build a matplotlib chart with a line graph showing births and deaths over time
## update the graph with locators for ticks on the x axis
years = mdates.YearLocator()
months = mdates.MonthLocator()
year_fmt = mdates.DateFormatter("%Y")

plt.figure(figsize=(14,8), dpi=200)
plt.title("Total Monthly Deaths and Births", fontsize=18)
plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14)

ax1 = plt.gca()
ax2 = ax1.twinx()

ax1.set_ylabel("Births", color="skyblue", fontsize=18)
ax2.set_ylabel("Deaths", color="crimson", fontsize=18)

ax1.set_xlim([df_monthly.date.min(), df_monthly.date.max()])
ax1.xaxis.set_major_locator(years)
ax1.xaxis.set_major_formatter(year_fmt)
ax1.xaxis.set_minor_locator(months)

ax1.grid(color="gray", linestyle="--")
ax1.plot(df_monthly.date, df_monthly.births, color="skyblue", linewidth=3)
ax2.plot(df_monthly.date, df_monthly.deaths, color="crimson", linewidth=3, linestyle="--")

plt.show()


## identify which clinic in the yearly data was busier
clin_1 = pd.DataFrame(df_yearly[df_yearly["clinic"] == "clinic 1"])
clin_2 = pd.DataFrame(df_yearly[df_yearly["clinic"] == "clinic 2"])

yearly_birth_line = px.line(df_yearly, x="year", y="births", color="clinic", title="Total Births Per Clinic")
yearly_birth_line.show()

yearly_death_line = px.line(df_yearly, x="year", y="deaths", color="clinic", title="Total deaths Per Clinic")
yearly_death_line.show()

df_yearly["pct_deaths"] = pd.to_numeric(df_yearly["deaths"]) /pd.to_numeric(df_yearly["births"])
yearly_death_pct_line = px.line(df_yearly, x="year", y="pct_deaths", color="clinic", title="Percentage of deaths to births")
yearly_death_pct_line.show()

## identify the start of hand washing
handwashing_start = pd.to_datetime('1847-06-01')
df_monthly["pct_deaths"] = pd.to_numeric(df_monthly["deaths"]) / pd.to_numeric(df_monthly["births"])

## calculate how much washing hands affected the death rate
before_wash = df_monthly[df_monthly["date"] < handwashing_start]
after_wash = df_monthly[df_monthly["date"] >= handwashing_start]

death_before_wash = before_wash["deaths"].sum() / before_wash["births"].sum() * 100
death_after_wash = after_wash["deaths"].sum() / after_wash["births"].sum() * 100

print(f"Average death rate before washing {death_before_wash:.3}")
print(f"Average deaths rate after washing {death_after_wash:.3}")

## set the rolling average prior to hand washing
rolling_df = before_wash.set_index("date")
rolling_df = rolling_df.rolling(window=6).mean()

## update the line graph to show percent deaths against the year
plt.figure(figsize=(14,8), dpi=200)
plt.title("Percentage of Monthly Deaths", fontsize=18)
plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14)

plt.ylabel("Percentagee of Deaths", color="crimson", fontsize=18)

ax = plt.gca()
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(year_fmt)
ax.xaxis.set_minor_locator(months)
ax.set_xlim([df_monthly["date"].min(), df_monthly["date"].max()])

plt.grid(color="gray", linestyle="--")

average_line, = plt.plot(rolling_df.index, rolling_df.pct_deaths, color="crimson", linewidth=3, label="6Month Rolling Ave")
before_wash_line, = plt.plot(before_wash.date, before_wash.pct_deaths, color="black", linewidth=1, linestyle="--", label="Before Handwashing")
after_wash_line, = plt.plot(after_wash.date, after_wash.pct_deaths, color="skyblue", linewidth=3, marker="o", label="After Handwashing")
plt.legend(handles=[average_line, before_wash_line, after_wash_line], fontsize=18)
plt.show()


## identify differences in average monthly death rates before and after hand washing
print(before_wash["deaths"].mean())
print(after_wash["deaths"].mean())

death_pct_bw = before_wash["deaths"].mean() * 100
death_pct_aw = after_wash["deaths"].mean() * 100
mean_delta = death_pct_bw - death_pct_aw
print(mean_delta)

ave_death_prob = death_pct_bw / death_pct_aw
print(ave_death_prob)

## make a box plot showing how the death rate changed due to washing hands
df_monthly["washing_hands"] = np.where(df_monthly.date < handwashing_start, "No", "Yes")

death_box = px.box(df_monthly, x="washing_hands", y="pct_deaths", color="washing_hands", title="How Did Hand Washing Change Death Statistics")
death_box.update_layout(xaxis_title="Washing Hands?", yaxis_title="Percentage of Monthly Deaths")
death_box.show()

## make a histogram showing the death rate changing due to hand washing
death_hist = px.histogram(df_monthly, x="pct_deaths", color="washing_hands", nbins=50, opacity=0.5, barmode="overlay", histnorm="percent", marginal="box")
death_hist.update_layout(xaxis_title="Proportion of Monthly Deaths", yaxis_title="Count")
death_hist.show()

## use seaborn kernel density estimate to more smoothly visualize distribution
plt.figure(dpi=200)
sns.kdeplot(before_wash.pct_deaths, fill=True)
sns.kdeplot(after_wash.pct_deaths, fill=True)
plt.title("Estimate Distribution of monthly death rate before and after hand washing")
plt.show()

## update the graph to show only positive death rates, not the erronious negative death rate
plt.figure(dpi=200)
sns.kdeplot(before_wash.pct_deaths, fill=True, clip=(0,100))
sns.kdeplot(after_wash.pct_deaths, fill=True, clip=(0,100))
plt.title("Estimate Distribution of monthly death rate before and after hand washing")
plt.xlim(0, 40)
plt.show()


## use a t test to show significant change
t_stat, p_value = stats.ttest_ind(a=before_wash.pct_deaths, b=after_wash.pct_deaths)
print(f"p value: {p_value:.9}")
print(f"t stat: {t_stat:.5}")