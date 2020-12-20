import base64
import pandas as pd
import streamlit as st
from pycaret.arules import *

st.set_page_config(
    page_title="Ruler", page_icon="📏", layout="centered", initial_sidebar_state="auto"
)

st.title("Ruler 📏")

"""
Using machine learning, Ruler will identify underlying relationtips between items and generate rules for them automagically 😎.

For more info about rule learning, see Association [Rule Learning](https://en.wikipedia.org/wiki/Association_rule_learning).
"""


st.markdown("---")
hints = st.empty()


# load data
@st.cache(allow_output_mutation=True, show_spinner=False)
def load_data(data):
    """
    try to load data in csv using pandas read_csv method, otherwise try in excel
    """
    try:
        return pd.read_csv(data)
    except:
        return pd.read_excel(data)


upload = st.file_uploader("upload your data in csv or xlsx", type=["csv", "xlsx"])

if upload is None:
    col1, col2, col3 = st.beta_columns(3)

    with col2:
        st.write("*or use the sample data below*")


with st.beta_expander("Data", expanded=True):
    with st.spinner("Epicly loading data..."):
        try:
            data = load_data(upload)
            # data = data.dropna(inplace=True)
            data
        except:
            data = load_data("data.csv")
            # data = data.dropna(inplace=True)
            data

        st.write(f"###### {data.shape[0]} rows and {data.shape[1]} columns")

# label columns
###########################################################

st.title("Parameter Selection")

st.subheader("What's what?")

col1, col2, col3, col4 = st.beta_columns(4)

with col1:
    item = st.selectbox("Item description column", options=(data.columns), index=2)

    item_hint = st.empty()

    try:
        st.write(f"* {data[item].nunique()} unique items")
    except:
        pass

with col2:
    tx = st.selectbox("Transaction column", options=(data.columns))

    tx_hint = st.empty()

    try:
        st.write(f"* {data[tx].nunique()} unique transactions")
        st.write(f"* {data[item].count()/data[tx].nunique():.1f} avg items/transaction")
    except:
        pass

with col3:
    confidence = st.number_input(
        "Confidence greater/equal to", value=0.85, min_value=0.0, max_value=1.0
    )

    confidence_hint = st.empty()

with col4:
    cols = data.columns.tolist()
    cols.insert(0, "None")

    date = st.selectbox("Date column (optional)", options=cols, index=5)

    date_hint = st.empty()

    @st.cache(allow_output_mutation=True)
    def to_datetime(column):
        """
        convert a column in a pandas dataframe to a pandas datetime format
        """
        data[column] = pd.to_datetime(data[column])
        return data[column]

    if date != "None":
        try:
            to_datetime(date)

            f"""
            * {data[date].min()}

            to

            * {data[date].max()}
            """
        except:
            st.warning(
                "⚠️ Smol oopsie, items in this column are in an invalid date format. If you don't have a dates, please select 'None'"
            )


if st.checkbox("Any items you want to ignore?"):
    ignore = st.multiselect("Any items you want to ignore?", options=data[item])
else:
    ignore = None

ignore_hint = st.empty()


st.markdown("---")
# st.title("Ruling")

get_rules = st.button("Get rules")  # <- leave if everything goes wrong


@st.cache(show_spinner=False)
def model_rules():
    """
    combine set up and create_model in one cached function. for some reason pycaret's built in combined function to do this (get_rules) doesn't work in streamlit
    """

    rule_setup = setup(data=data, transaction_id=tx, item_id=item, ignore_items=ignore)

    rules = create_model(threshold=confidence)

    return rules


# session state for get_rules button
###########################################################
import SessionState

session_state = SessionState.get(get_rules=None)

if get_rules or session_state.get_rules:
    session_state.get_rules = True
    with st.spinner("Ruling the rules..."):
        rules = model_rules()


def format_rules(df):

    """
    filter a few columns and sort by confidence
    """

    df = df.filter(items=["antecedents", "consequents", "confidence", "support"])

    df.sort_values(by=["confidence"], ascending=False)

    # change from frozenset{} to []
    # slight modification of https://discuss.streamlit.io/t/problem-with-how-a-df-is-being-displayed/7218/4
    # when using tuple() there was a weird comma like ['item',]
    df = df.applymap(lambda x: list(x) if isinstance(x, frozenset) else x)

    # this regex is only needed if the frozenset part is in the dataframe as a string
    # like if you save and read using pandas' to_csv and thenr read_csv
    # rules["antecedents"] = rules["antecedents"].str.extract(r"\{(.*?)\}")
    # rules["consequents"] = rules["consequents"].str.extract(r"\{(.*?)\}")

    return df


if get_rules or session_state.get_rules:

    st.title("Results 📋")

    download_link = st.empty()
    st.markdown("---")

    st.write(f"Number of rules = {len(rules)}")

    rules = format_rules(df=rules)

    table_hint = st.empty()

    st.table(rules)

    # generate data summary for optional download

    # summary = pd.DataFrame(
    #     {
    #         "unique items": data[item].nunique(),
    #         "unique transactions": data[tx].nunique(),
    #         "avg items/transaction": f"{data[item].count() / data[tx].nunique():.1f}",
    #         "date range": f"{data[date].min()} to {data[date].max()}",
    #         "item column used": item,
    #         "transaction column used": tx,
    #         "confidence used": confidence,
    #         "number of rules": len(rules),
    #     },
    #     index=[0],
    # )
    # ^ horizonal didn't look good

    if date != "None":

        summary = pd.DataFrame(
            {
                "Feature": [
                    "unique items",
                    "unique transactions",
                    "avg items/transaction",
                    "date range",
                    "item column used",
                    "transaction column used",
                    "confidence used",
                    "number of rules",
                ],
                "Info": [
                    data[item].nunique(),
                    data[tx].nunique(),
                    f"{data[item].count() / data[tx].nunique():.1f}",
                    f"{data[date].min()} to {data[date].max()}",
                    item,
                    tx,
                    confidence,
                    len(rules),
                ],
            }
        )

    else:
        summary = pd.DataFrame(
            {
                "Feature": [
                    "unique items",
                    "unique transactions",
                    "avg items/transaction",
                    "date range",
                    "item column used",
                    "transaction column used",
                    "confidence used",
                    "number of rules",
                ],
                "Info": [
                    data[item].nunique(),
                    data[tx].nunique(),
                    f"{data[item].count() / data[tx].nunique():.1f}",
                    "no date column selected",
                    item,
                    tx,
                    confidence,
                    len(rules),
                ],
            }
        )

    # st.write(type(rules))
    # with pd.ExcelWriter("ruler_report.xlsx", engine="xlsxwriter") as writer:
    #     data.to_excel(writer, sheet_name="input_data", index=False)
    #     summary.to_excel(writer, sheet_name="summary_of_stats", index=False)
    #     rules.to_excel(writer, sheet_name="rules", index=False)

    # ModuleNotFoundError: No module named 'openpyxl'
    # had to pip install openpyxl for this to work
    # also
    # ModuleNotFoundError: No module named 'xlsxwriter'
    # pip install xlsxwriter
    # gave up, too many errors :(

    # for download_link
    def filedownload(data, text="Download as CSV", file_name="data.csv"):
        csv = data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download={file_name}>{text}</a>'
        return href

    # attempted modification to download the excel file with all the sheets
    # def filedownload(data, text="Download as XLSX", file_name="data.csv"):
    #     excel = pd.read_excel(data)
    #     b64 = base64.b64encode(excel.encode()).decode()  # strings <-> bytes conversions
    #     href = f'<a href="data:file/excel;base64,{b64}" download={file_name}>{text}</a>'
    #     return href

    # download_link.markdown(
    #     filedownload(data="ruler_report.xlsx", file_name="ruler_report.xlsx"),
    #     unsafe_allow_html=True,
    # )

    # creating download links for rules, input data and the summary
    # had to resort to this since couldn't get excel thing working
    with download_link.beta_container():

        col1, col2, col3 = st.beta_columns(3)

        with col1:
            st.markdown(
                filedownload(data=rules, file_name="rules.csv", text="Download rules"),
                unsafe_allow_html=True,
            )

            st.write("###### The generated rules you see below")

        with col2:
            st.markdown(
                filedownload(
                    data=summary, file_name="summary.csv", text="Download summary"
                ),
                unsafe_allow_html=True,
            )

            st.write("###### Stats of input data, rules and settings used")

        with col3:
            st.markdown(
                filedownload(
                    data=data, file_name="rules_report.csv", text="Download input data"
                ),
                unsafe_allow_html=True,
            )

            st.write("###### Input data")

# hints
###########################################################

if hints.checkbox("Click here to show hints 😊"):
    tx_hint.info(
        "###### **💡 Hint**: unique identifier, like invoice number that shows items per transaction"
    )

    item_hint.info(
        "###### **💡 Hint**: contains the names of the items you want to find rules for"
    )

    confidence_hint.info("###### **💡 Hint**: filter rules by confidence")

    ignore_hint.info(
        "###### **💡 Hint**: things in your item list that might not be useful like delivery charges"
    )

    if get_rules or session_state.get_rules:
        st.markdown(
            table_hint.info(
                """

                ###### **💡 Hint:**

                * ###### If **antecedents** are A, B and **consequents** are C - it means people who buy A and B also frequently buy C

                * ###### **Confidence** is an indication of how often the rule has been found to be true

                * ###### **Support** is an indication of how frequently the itemset appears in the dataset
                """
            )
        )