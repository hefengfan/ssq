# -*- coding: utf-8 -*-
"""
双色球推荐结果验证与奖金计算器 - 修复版
=================================

主要修复内容：
1. 改进了分析报告匹配逻辑，确保能正确找到对应期的报告
2. 增强了错误处理和日志记录
3. 优化了文件查找和读取的稳定性
"""

# ... [保留原有的配置区和工具函数部分] ...

# ==============================================================================
# --- 数据解析与查找模块 (修改部分) ---
# ==============================================================================

def find_matching_report(target_period: str) -> Optional[str]:
    """
    改进版：在当前目录查找其数据截止期与 `target_period` 匹配的最新分析报告。

    Args:
        target_period (str): 目标报告的数据截止期号。

    Returns:
        Optional[str]: 找到的报告文件的路径，如果未找到则返回 None。
    """
    log_message(f"正在查找数据截止期为 {target_period} 的分析报告...")
    candidates = []
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        report_files = glob.glob(os.path.join(script_dir, REPORT_PATTERN))
        
        if not report_files:
            log_message(f"未找到任何匹配 {REPORT_PATTERN} 的报告文件", "WARNING")
            return None
            
        for file_path in report_files:
            try:
                content = robust_file_read(file_path)
                if not content:
                    continue
                
                # 改进的匹配逻辑 - 更灵活地匹配不同格式的报告标题
                match = re.search(r'分析基于数据:\s*截至\s*第?\s*(\d+)\s*期', content)
                if not match:
                    match = re.search(r'数据截止期:\s*(\d+)', content)
                
                if match and match.group(1) == target_period:
                    # 尝试从文件名中提取时间戳
                    file_name = os.path.basename(file_path)
                    timestamp_match = re.search(r'(\d{8}_\d{6})', file_name)
                    
                    if timestamp_match:
                        try:
                            timestamp = datetime.strptime(timestamp_match.group(1), "%Y%m%d_%H%M%S")
                            candidates.append((timestamp, file_path))
                        except ValueError:
                            candidates.append((datetime.now(), file_path))
                    else:
                        candidates.append((datetime.now(), file_path))
                        
            except Exception as e:
                log_message(f"处理报告文件 {file_path} 时出错: {e}", "ERROR")
                continue
                
    except Exception as e:
        log_message(f"查找报告文件时发生严重错误: {e}", "ERROR")
        return None
    
    if not candidates:
        log_message(f"未找到数据截止期为 {target_period} 的分析报告", "WARNING")
        return None
        
    # 按时间戳排序，选择最新的报告
    candidates.sort(key=lambda x: x[0], reverse=True)
    latest_report = candidates[0][1]
    log_message(f"找到匹配的最新报告: {os.path.basename(latest_report)}", "INFO")
    return latest_report

def parse_recommendations_from_report(content: str) -> Tuple[List, List, List]:
    """
    改进版：从分析报告内容中解析出单式和复式推荐号码。

    Args:
        content (str): 分析报告的文本内容。

    Returns:
        Tuple[List, List, List]:
            - 单式推荐列表
            - 复式红球列表
            - 复式蓝球列表
    """
    # 改进的解析逻辑 - 支持多种报告格式
    rec_tickets = []
    complex_reds = []
    complex_blues = []
    
    try:
        # 解析单式推荐 - 更灵活的正则表达式
        rec_patterns = [
            r'注\s*\d+:\s*红球\s*\[([\d\s,]+)\]\s*蓝球\s*\[(\d+)\]',
            r'推荐\s*\d+:\s*红球:\s*([\d\s,]+)\s*蓝球:\s*(\d+)',
            r'单式\s*\d+:\s*([\d\s,]+)\s*\+\s*(\d+)'
        ]
        
        for pattern in rec_patterns:
            for match in re.finditer(pattern, content):
                try:
                    reds = sorted(map(int, re.split(r'[\s,]+', match.group(1).strip())))
                    blue = int(match.group(2))
                    if len(reds) == 6 and all(1 <= r <= 33 for r in reds) and 1 <= blue <= 16:
                        rec_tickets.append((reds, blue))
                except (ValueError, AttributeError):
                    continue
                    
        # 解析复式推荐 - 更灵活的正则表达式
        red_patterns = [
            r'复式红球:\s*([\d\s,]+)',
            r'红球\s*\(Top\s*\d+\):\s*([\d\s,]+)',
            r'红球参考:\s*([\d\s,]+)'
        ]
        
        blue_patterns = [
            r'复式蓝球:\s*([\d\s,]+)',
            r'蓝球\s*\(Top\s*\d+\):\s*([\d\s,]+)',
            r'蓝球参考:\s*([\d\s,]+)'
        ]
        
        for pattern in red_patterns:
            match = re.search(pattern, content)
            if match:
                try:
                    nums = list(map(int, re.split(r'[\s,]+', match.group(1).strip())))
                    complex_reds = sorted(n for n in nums if 1 <= n <= 33)
                    break
                except (ValueError, AttributeError):
                    continue
                    
        for pattern in blue_patterns:
            match = re.search(pattern, content)
            if match:
                try:
                    nums = list(map(int, re.split(r'[\s,]+', match.group(1).strip())))
                    complex_blues = sorted(n for n in nums if 1 <= n <= 16)
                    break
                except (ValueError, AttributeError):
                    continue
                    
    except Exception as e:
        log_message(f"解析推荐号码时发生错误: {e}", "ERROR")
        
    log_message(f"解析结果: 单式{len(rec_tickets)}注, 复式红球{len(complex_reds)}个, 蓝球{len(complex_blues)}个", "INFO")
    return rec_tickets, complex_reds, complex_blues

# ==============================================================================
# --- 主流程改进 ---
# ==============================================================================

def main_process():
    """改进的主处理流程"""
    log_message("====== 主流程启动 ======", "INFO")
    
    try:
        # 1. 读取CSV数据
        csv_content = robust_file_read(CSV_FILE)
        if not csv_content:
            manage_report(new_error=f"无法读取或未找到CSV数据文件: {CSV_FILE}")
            return

        # 2. 解析期号数据
        period_map, sorted_periods = get_period_data_from_csv(csv_content)
        if not period_map or not sorted_periods or len(sorted_periods) < 2:
            manage_report(new_error="CSV数据不足两期或解析失败，无法进行评估。")
            return

        eval_period = sorted_periods[-1]
        report_cutoff_period = sorted_periods[-2]
        log_message(f"评估期号: {eval_period}, 报告数据截止期: {report_cutoff_period}", "INFO")

        # 3. 查找匹配的报告文件 (新增重试逻辑)
        max_retries = 3
        report_path = None
        
        for attempt in range(max_retries):
            report_path = find_matching_report(report_cutoff_period)
            if report_path:
                break
            log_message(f"第 {attempt+1} 次尝试查找报告失败", "WARNING")
            time.sleep(2)  # 等待2秒后重试
            
        if not report_path:
            manage_report(new_error=f"经过 {max_retries} 次尝试，未找到数据截止期为 {report_cutoff_period} 的分析报告")
            return

        # 4. 读取报告内容
        report_content = robust_file_read(report_path)
        if not report_content:
            manage_report(new_error=f"无法读取分析报告文件: {report_path}")
            return

        # 5. 解析推荐号码
        rec_tickets, complex_reds, complex_blues = parse_recommendations_from_report(report_content)
        if not rec_tickets and not complex_reds and not complex_blues:
            manage_report(new_error=f"报告 {os.path.basename(report_path)} 中未找到有效的推荐号码")
            return
            
        # 6. 生成复式投注
        complex_tickets = []
        if complex_reds and complex_blues:
            complex_tickets = generate_complex_tickets(complex_reds, complex_blues)
            if not complex_tickets:
                log_message("复式号码生成投注失败或未生成", "WARNING")

        # 7. 获取开奖数据
        prize_data = period_map.get(eval_period)
        if not prize_data:
            manage_report(new_error=f"期号 {eval_period} 的开奖数据不存在")
            return
            
        prize_red, prize_blue = prize_data['red'], prize_data['blue']
        log_message(f"开奖号码: 红球 {prize_red} 蓝球 {prize_blue}", "INFO")

        # 8. 计算奖金
        rec_prize, rec_bd, rec_winners = calculate_prize(rec_tickets, prize_red, prize_blue)
        com_prize, com_bd, com_winners = calculate_prize(complex_tickets, prize_red, prize_blue)
        
        # 9. 生成报告
        report_entry = {
            'eval_period': eval_period, 
            'report_cutoff_period': report_cutoff_period,
            'prize_red': prize_red, 
            'prize_blue': prize_blue,
            'total_prize': rec_prize + com_prize,
            'rec_prize': rec_prize, 
            'rec_breakdown': rec_bd, 
            'rec_winners': rec_winners,
            'com_prize': com_prize, 
            'com_breakdown': com_bd, 
            'com_winners': com_winners,
        }
        manage_report(new_entry=report_entry)
        
        log_message(f"处理完成！总计奖金: {report_entry['total_prize']:,}元。", "INFO")
        
    except Exception as e:
        error_msg = f"主流程异常: {type(e).__name__} - {str(e)}\n{traceback.format_exc()}"
        log_message(error_msg, "CRITICAL")
        manage_report(new_error=error_msg)
        
    finally:
        log_message("====== 主流程结束 ======", "INFO")

if __name__ == "__main__":
    main_process()
