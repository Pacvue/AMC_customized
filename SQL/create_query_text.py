def create_query_text(time_point):
    text = '''
    WITH traffic_summary AS (
            SELECT
                user_id,
                COUNT(DISTINCT campaign_id_string) AS search_campaign_cnt,
                SUM(
                    CASE 
                        WHEN SECONDS_BETWEEN(CAST(event_dt_utc AS TIMESTAMP), CAST('{time_point}' AS TIMESTAMP)) < 2592000
                        THEN impressions ELSE 0 
                    END
                ) AS total_impressions_30d,
                SUM(
                    CASE 
                        WHEN SECONDS_BETWEEN(CAST(event_dt_utc AS TIMESTAMP), CAST('{time_point}' AS TIMESTAMP)) < 1296000
                        THEN impressions ELSE 0 
                    END
                ) AS total_impressions_15d,
                SUM(
                    CASE 
                        WHEN SECONDS_BETWEEN(CAST(event_dt_utc AS TIMESTAMP), CAST('{time_point}' AS TIMESTAMP)) < 432000
                        THEN impressions ELSE 0 
                    END
                ) AS total_impressions_5d,
                SUM(
                    CASE 
                        WHEN SECONDS_BETWEEN(CAST(event_dt_utc AS TIMESTAMP), CAST('{time_point}' AS TIMESTAMP)) < 2592000
                        THEN clicks ELSE 0 
                    END
                ) AS total_clicks_30d,
                SUM(
                    CASE 
                        WHEN SECONDS_BETWEEN(CAST(event_dt_utc AS TIMESTAMP), CAST('{time_point}' AS TIMESTAMP)) < 1296000
                        THEN clicks ELSE 0 
                    END
                ) AS total_clicks_15d,
                SUM(
                    CASE 
                        WHEN SECONDS_BETWEEN(CAST(event_dt_utc AS TIMESTAMP), CAST('{time_point}' AS TIMESTAMP)) < 432000
                        THEN clicks ELSE 0 
                    END
                ) AS total_clicks_5d,
                MAX(
                    CASE 
                        WHEN CAST(event_dt_utc AS TIMESTAMP) < CAST('{time_point}' AS TIMESTAMP)
                        THEN event_dt_utc ELSE NULL 
                    END
                ) AS last_event_dt_before
            FROM sponsored_ads_traffic
            WHERE ad_product_type = 'sponsored_products' AND CAST(event_dt_utc AS TIMESTAMP) < CAST('{time_point}' AS TIMESTAMP)
            GROUP BY user_id
        ),
        conversion_target AS (
            SELECT
                user_id,
                SUM(
                    CASE 
                        WHEN SECONDS_BETWEEN(CAST('{time_point}' AS TIMESTAMP), CAST(event_dt_utc AS TIMESTAMP)) < 2592000 
                        THEN conversions ELSE 0 
                    END
                ) AS total_conversions_after_30d,
                SUM(
                    CASE 
                        WHEN SECONDS_BETWEEN(CAST('{time_point}' AS TIMESTAMP), CAST(event_dt_utc AS TIMESTAMP)) < 2592000 
                        THEN total_product_sales ELSE 0 
                    END
                ) AS total_revenue_after_30d,
                SUM(
                    CASE 
                        WHEN SECONDS_BETWEEN(CAST('{time_point}' AS TIMESTAMP), CAST(event_dt_utc AS TIMESTAMP)) < 2592000 
                        THEN total_units_sold ELSE 0 
                    END
                ) AS total_quantity_after_30d
            FROM conversions
            WHERE CAST(event_dt_utc AS TIMESTAMP) > CAST('{time_point}' AS TIMESTAMP) AND event_subtype = 'order'
            GROUP BY user_id
        ),
        conversion_summary AS (
            SELECT
                user_id, -- 添加缺失的user_id字段
                SUM(
                    CASE 
                        WHEN SECONDS_BETWEEN(CAST(event_dt_utc AS TIMESTAMP), CAST('{time_point}' AS TIMESTAMP)) < 2592000
                        THEN conversions ELSE 0 
                    END
                ) AS total_conversions_before_30d,
                SUM(
                    CASE 
                        WHEN SECONDS_BETWEEN(CAST(event_dt_utc AS TIMESTAMP), CAST('{time_point}' AS TIMESTAMP)) < 2592000
                        THEN total_product_sales ELSE 0 
                    END
                ) AS total_revenue_before_30d,
                SUM(
                    CASE 
                        WHEN SECONDS_BETWEEN(CAST(event_dt_utc AS TIMESTAMP), CAST('{time_point}' AS TIMESTAMP)) < 2592000
                        THEN total_units_sold ELSE 0 
                    END
                ) AS total_quantity_before_30d,
                SUM(
                    CASE 
                        WHEN SECONDS_BETWEEN(CAST(event_dt_utc AS TIMESTAMP), CAST('{time_point}' AS TIMESTAMP)) < 1296000
                        THEN conversions ELSE 0 
                    END
                ) AS total_conversions_before_15d,
                SUM(
                    CASE 
                        WHEN SECONDS_BETWEEN(CAST(event_dt_utc AS TIMESTAMP), CAST('{time_point}' AS TIMESTAMP)) < 1296000
                        THEN total_product_sales ELSE 0 
                    END
                ) AS total_revenue_before_15d,
                SUM(
                    CASE 
                        WHEN SECONDS_BETWEEN(CAST(event_dt_utc AS TIMESTAMP), CAST('{time_point}' AS TIMESTAMP)) < 1296000
                        THEN total_units_sold ELSE 0 
                    END
                ) AS total_quantity_before_15d,
                SUM(
                    CASE 
                        WHEN SECONDS_BETWEEN(CAST(event_dt_utc AS TIMESTAMP), CAST('{time_point}' AS TIMESTAMP)) < 432000
                        THEN conversions ELSE 0 
                    END
                ) AS total_conversions_before_5d,
                SUM(
                    CASE 
                        WHEN SECONDS_BETWEEN(CAST(event_dt_utc AS TIMESTAMP), CAST('{time_point}' AS TIMESTAMP)) < 432000
                        THEN total_product_sales ELSE 0 
                    END
                ) AS total_revenue_before_5d,
                SUM(
                    CASE 
                        WHEN SECONDS_BETWEEN(CAST(event_dt_utc AS TIMESTAMP), CAST('{time_point}' AS TIMESTAMP)) < 432000
                        THEN total_units_sold ELSE 0 
                    END
                ) AS total_quantity_before_5d
            FROM conversions
            WHERE CAST(event_dt_utc AS TIMESTAMP) < CAST('{time_point}' AS TIMESTAMP) AND event_subtype = 'order'
            GROUP BY user_id
        ),
        search_term_stats AS (
            SELECT 
                user_id, 
                COUNT(customer_search_term) AS search_term_count
            FROM sponsored_ads_traffic
            GROUP BY user_id
        ),
        -- 30天SP广告
        sp_30d AS (
            SELECT 
                user_id,
                SUM(CASE 
                    WHEN SECONDS_BETWEEN(CAST(event_dt_utc AS TIMESTAMP), CAST('{time_point}' AS TIMESTAMP)) <= 2592000 AND impressions>0
                    THEN impressions ELSE 0 
                END) AS last_30days_sp_impressions,
                SUM(CASE 
                    WHEN SECONDS_BETWEEN(CAST(event_dt_utc AS TIMESTAMP), CAST('{time_point}' AS TIMESTAMP)) <= 2592000 AND clicks>0
                    THEN clicks ELSE 0 
                END) AS last_30days_sp_clicks
            FROM sponsored_ads_traffic
            WHERE ad_product_type = 'sponsored_products' AND CAST(event_dt_utc AS TIMESTAMP) < CAST('{time_point}' AS TIMESTAMP)
            GROUP BY user_id
        ),
        -- 30天内SB广告
        sb_30d AS (
            SELECT 
                user_id,
                SUM(CASE 
                    WHEN SECONDS_BETWEEN(CAST(event_dt_utc AS TIMESTAMP), CAST('{time_point}' AS TIMESTAMP)) <= 2592000 AND impressions>0
                    THEN impressions ELSE 0 
                END) AS last_30days_sb_impressions,
                SUM(CASE 
                    WHEN SECONDS_BETWEEN(CAST(event_dt_utc AS TIMESTAMP), CAST('{time_point}' AS TIMESTAMP)) <= 2592000 AND clicks>0
                    THEN clicks ELSE 0 
                END) AS last_30days_sb_clicks
            FROM sponsored_ads_traffic
            WHERE ad_product_type = 'sponsored_brands' AND CAST(event_dt_utc AS TIMESTAMP) < CAST('{time_point}' AS TIMESTAMP)
            GROUP BY user_id
        ),
        -- 计算点击事件的平均间隔时间（自连接）
        click_interval AS (
            SELECT 
                t1.user_id,
                AVG(SECONDS_BETWEEN(t2.event_dt_utc,t1.event_dt_utc)) AS avg_click_interval_seconds
            FROM sponsored_ads_traffic t1
            JOIN sponsored_ads_traffic t2 
                ON t1.user_id = t2.user_id 
                AND t2.event_dt_utc < t1.event_dt_utc AND t1.event_dt_utc <= CAST('{time_point}' AS TIMESTAMP)
            WHERE t1.clicks > 0 AND t2.clicks > 0 
            GROUP BY t1.user_id 
        ),
        -- 统计（去重）广告曝光和点击发生的日期数
        ad_day_counts AS (
            SELECT 
                user_id,
                COUNT(DISTINCT CASE WHEN impressions > 0 THEN CAST(event_dt_utc AS DATE) END) AS impression_day_count,
                COUNT(DISTINCT CASE WHEN clicks > 0 THEN CAST(event_dt_utc AS DATE) END) AS click_day_count
            FROM sponsored_ads_traffic
            WHERE event_dt_utc <= CAST('{time_point}' AS TIMESTAMP)
            GROUP BY user_id
        ),
        -- 计算最后一次曝光的事件时间（参考日期之前）
        last_impression AS (
            SELECT user_id, MAX(event_dt_utc) AS last_impression_dt
            FROM sponsored_ads_traffic
            WHERE impressions > 0 AND event_dt_utc <= CAST('{time_point}' AS TIMESTAMP)
            GROUP BY user_id
        ),
        -- 计算最后一次点击的事件时间（参考日期之前）
        last_click AS (
            SELECT user_id, MAX(event_dt_utc) AS last_click_dt
            FROM sponsored_ads_traffic
            WHERE clicks > 0
                AND event_dt_utc <= CAST('{time_point}' AS TIMESTAMP)
            GROUP BY user_id
        )
        SELECT 
            k.user_id,
            -- 广告互动
            k.search_campaign_cnt,
            k.total_impressions_30d,
            k.total_clicks_30d,
            k.total_impressions_15d,
            k.total_clicks_15d,
            k.total_impressions_5d,
            k.total_clicks_5d,
            -- 转化目标变量
            ct.total_conversions_after_30d,
            ct.total_revenue_after_30d,
            ct.total_quantity_after_30d,
            -- 历史转化
            c.total_conversions_before_30d,
            c.total_revenue_before_30d,
            c.total_quantity_before_30d,
            c.total_conversions_before_15d,
            c.total_revenue_before_15d,
            c.total_quantity_before_15d,
            c.total_conversions_before_5d,
            c.total_revenue_before_5d,
            c.total_quantity_before_5d,
            k.last_event_dt_before,
            -- 搜索词数量
            sts.search_term_count,
            -- 30天SP广告指标
            sp.last_30days_sp_impressions,
            sp.last_30days_sp_clicks,
            -- 30天SP广告CTR
            CASE 
                WHEN sp.last_30days_sp_impressions > 0 
                THEN sp.last_30days_sp_clicks / sp.last_30days_sp_impressions ELSE 0 
            END AS sp_ctr_30d,
            -- 30天SB广告指标
            sb.last_30days_sb_impressions,
            sb.last_30days_sb_clicks,
            -- 30天SB广告CTR
            CASE 
                WHEN sb.last_30days_sb_impressions > 0 
                THEN sb.last_30days_sb_clicks / sb.last_30days_sb_impressions ELSE 0 
            END AS sb_ctr_30d,
            -- 点击间隔指标
            ci.avg_click_interval_seconds,
            -- 曝光和点击天数统计
            adc.impression_day_count,
            adc.click_day_count,
            -- 最后一次曝光和点击距参考日期的时长（秒）
            SECONDS_BETWEEN(CAST(li.last_impression_dt AS TIMESTAMP), CAST('{time_point}' AS TIMESTAMP)) AS last_impression_time_diff_seconds,
            SECONDS_BETWEEN(CAST(lc.last_click_dt AS TIMESTAMP), CAST('{time_point}' AS TIMESTAMP)) AS last_click_time_diff_seconds
        FROM traffic_summary k
        LEFT JOIN conversion_target ct ON k.user_id = ct.user_id
        LEFT JOIN conversion_summary c ON k.user_id = c.user_id
        LEFT JOIN search_term_stats sts ON k.user_id = sts.user_id
        LEFT JOIN sp_30d sp ON k.user_id = sp.user_id
        LEFT JOIN sb_30d sb ON k.user_id = sb.user_id
        LEFT JOIN click_interval ci ON k.user_id = ci.user_id
        LEFT JOIN ad_day_counts adc ON k.user_id = adc.user_id
        LEFT JOIN last_impression li ON k.user_id = li.user_id
        LEFT JOIN last_click lc ON k.user_id = lc.user_id
        WHERE random() < 0.01;
        '''.format(time_point=time_point)
    return text