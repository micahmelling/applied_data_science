import streamlit as st
import pandas as pd


def accept_ad_target_group():
    return st.selectbox(
        'ad_target_group',
        ('level_1', 'level_2', 'level_3', 'level_4', 'level_5'))


def accept_marketing_message():
    return st.selectbox(
        'marketing_message',
        ('level_1', 'level_2', 'level_3'))


def accept_xp_points():
    return st.text_input('xp_points')


def accept_activity_score():
    return st.text_input('activity_score')


def create_entry_dataframe(ad_target_group, marketing_message, xp_point, activity_score):
    df = pd.DataFrame({
        'ad_target_group': [ad_target_group],
        'marketing_message': [marketing_message],
        'xp_points': [xp_point],
        'activity_score': [activity_score]
    })
    return df


def create_prediction_data_skeleton_df():
    columns = [
        'activity_score', 'propensity_score', 'profile_score_new', 'completeness_score', 'xp_points',
        'profile_score', 'portfolio_score', 'mouse_movement', 'average_stars', 'ad_target_group',
        'marketing_message', 'device_type', 'all_star_group', 'mouse_x', 'coupon_code', 'ad_engagement_group',
        'user_group', 'browser_type', 'email_code', 'marketing_creative', 'secondary_user_group',
        'promotion_category', 'marketing_campaign', 'mouse_y', 'marketing_channel', 'marketing_creative_sub',
        'site_level', 'acquired_date'
    ]
    df = pd.DataFrame(columns=columns)
    return df


def create_prediction_dataframe(prediction_skeleton_df, entry_df):
    return pd.concat([prediction_skeleton_df, entry_df], axis=0)


def get_prediction():
    pass


def show_shap_values():
    pass


def main():
    st.title('CHURN MODEL UI')
    st.text('''Enter the scenario for which you want a prediction.''')
    ad_target_group = accept_ad_target_group()
    marketing_message = accept_marketing_message()
    xp_points = accept_xp_points()
    activity_score = accept_activity_score()
    button_sent = st.button('SUBMIT CHANGES')
    if button_sent:
        pass


if __name__ == "__main__":
    main()

