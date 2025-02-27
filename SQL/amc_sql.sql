-- WITH purchase_summary AS (
--     SELECT
--         user_id,
--         MAX(event_date_utc) AS last_purchase_date,
--         COUNT(user_id) AS total_purchases,
--         SUM(total_product_sales) AS total_sales_value,
--         AVG(total_product_sales / COALESCE(total_units_sold, 0)) AS avg_order_value
--     FROM conversions_for_audiences
--     WHERE event_subtype = 'order'
--     GROUP BY user_id
-- ),
-- engagement_summary AS (
--     SELECT
--         user_id,
--         SUM(impressions) AS total_impressions,
--         SUM(clicks) AS total_clicks,
--         COUNT(DISTINCT impression_dt_utc) AS active_days,
--         COUNT(DISTINCT campaign_id) AS distinct_campaigns
--     FROM (
--         SELECT user_id, impressions, 0 AS clicks, impression_dt_utc, campaign_id
--         FROM dsp_impressions_for_audiences
--         UNION ALL
--         SELECT user_id, 0 AS impressions, clicks, click_dt_utc, campaign_id
--         FROM dsp_clicks_for_audiences
--     ) engagement_data
--     GROUP BY user_id
-- )
-- SELECT
--     ps.user_id,
--     SECONDS_BETWEEN(ps.last_purchase_date, CAST('today' AS DATE)) AS recency,
--     ps.total_purchases AS frequency,
--     ps.total_sales_value AS monetary,
--     ps.avg_order_value,
--     es.total_impressions,
--     es.total_clicks,
--     (es.total_clicks * 1.0 / COALESCE(es.total_impressions, 0)) AS click_through_rate,
--     es.active_days,
--     es.distinct_campaigns
-- FROM purchase_summary ps
-- LEFT JOIN engagement_summary es ON ps.user_id = es.user_id


-- WITH purchase_summary AS 
-- (SELECT user_id, MAX(event_date_utc) AS last_purchase_date, 
-- COUNT(user_id) AS total_purchases, 
-- SUM(total_product_sales) AS total_sales_value, 
-- AVG(total_product_sales / COALESCE(total_units_sold, 0)) AS avg_order_value 
-- FROM conversions_for_audiences WHERE event_subtype = 'order' GROUP BY user_id ),

--  engagement_summary AS 
--  (SELECT user_id, SUM(impressions) AS total_impressions, SUM(clicks) AS total_clicks, 
--  COUNT(DISTINCT impression_dt_utc) AS active_days, 
--  COUNT(DISTINCT campaign_id) AS distinct_campaigns 
--  FROM 
 
--  (SELECT user_id, impressions, 0 AS clicks, impression_dt_utc, campaign_id 
--  FROM dsp_impressions_for_audiences 
--  UNION ALL SELECT user_id, 0 AS impressions, clicks, click_dt_utc, campaign_id 
--  FROM dsp_clicks_for_audiences ) engagement_data 
--  GROUP BY user_id ) 
 
--  SELECT ps.user_id, SECONDS_BETWEEN(ps.last_purchase_date, CAST('today' AS DATE)) AS recency, 
--  ps.total_purchases AS frequency, ps.total_sales_value AS monetary, 
--  ps.avg_order_value, es.total_impressions, es.total_clicks,
--  (es.total_clicks * 1.0 / COALESCE(es.total_impressions, 0)) AS click_through_rate,
--   es.active_days, es.distinct_campaigns 
  
--   FROM purchase_summary ps 
--   LEFT JOIN engagement_summary es 
--   ON ps.user_id = es.user_id



WITH purchase_summary AS (SELECT user_id, MAX(event_date_utc) AS last_purchase_date, COUNT(user_id) AS total_purchases, 

SUM(total_product_sales) AS total_sales_value, AVG(total_product_sales / COALESCE(total_units_sold, 0)) AS avg_order_value 

FROM conversions WHERE event_subtype = 'order' GROUP BY user_id ), 

engagement_summary AS (SELECT user_id, SUM(impressions) AS total_impressions, SUM(clicks) AS total_clicks, 

COUNT(DISTINCT impression_dt_utc) AS active_days, COUNT(DISTINCT campaign_id) AS distinct_campaigns 

FROM (SELECT user_id, impressions, 0 AS clicks, impression_dt_utc, campaign_id 

FROM dsp_impressions UNION ALL SELECT user_id, 0 AS impressions, clicks, click_dt_utc, campaign_id 

FROM dsp_clicks ) engagement_data GROUP BY user_id ) 

SELECT ps.user_id, SECONDS_BETWEEN(ps.last_purchase_date, CAST('today' AS DATE)) 

AS recency, ps.total_purchases AS frequency, ps.total_sales_value AS monetary, 

ps.avg_order_value, es.total_impressions, es.total_clicks, 

(es.total_clicks * 1.0 / COALESCE(es.total_impressions, 0)) AS click_through_rate, es.active_days, es.distinct_campaigns 

FROM purchase_summary ps LEFT JOIN engagement_summary es ON ps.user_id = es.user_id