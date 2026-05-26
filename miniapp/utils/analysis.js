function pickNumber() {
  for (let index = 0; index < arguments.length; index += 1) {
    const value = arguments[index];
    if (typeof value === 'number' && Number.isFinite(value)) {
      return Math.round(value * 10) / 10;
    }
    if (typeof value === 'string' && value.trim()) {
      const parsed = Number(value);
      if (Number.isFinite(parsed)) {
        return Math.round(parsed * 10) / 10;
      }
    }
  }
  return null;
}

function pickText() {
  for (let index = 0; index < arguments.length; index += 1) {
    const value = arguments[index];
    if (typeof value === 'string' && value.trim()) {
      return value.trim();
    }
  }
  return '暂无护理建议';
}

function pickRawText() {
  for (let index = 0; index < arguments.length; index += 1) {
    const value = arguments[index];
    if (typeof value === 'string' && value.trim()) {
      return value.trim();
    }
  }
  return '';
}

function scoreText(score) {
  if (score === null || typeof score === 'undefined') {
    return '--';
  }
  return String(score);
}

function normalizeAnalysisEnvelope(payload) {
  const result = payload && typeof payload === 'object' ? payload : {};
  if (
    result.analysis_result &&
    typeof result.analysis_result === 'object' &&
    (result.analysis_type || result.status === 'success')
  ) {
    return {
      analysisType: result.analysis_type || result.analysis_result.analysis_type || '',
      status: result.status || result.analysis_result.status || '',
      taskResult: result,
      analysisResult: result.analysis_result,
      alignResult: result.align_result || null,
    };
  }

  return {
    analysisType: result.analysis_type || '',
    status: result.status || '',
    taskResult: null,
    analysisResult: result,
    alignResult: result.align_result || null,
  };
}

function isTongueAnalysis(result) {
  const envelope = normalizeAnalysisEnvelope(result);
  const analysis = envelope.analysisResult || {};
  return (
    envelope.analysisType === 'tongue' ||
    Boolean(analysis.tongue_color) ||
    Boolean(analysis.tongue_moisture) ||
    Boolean(analysis.demo_report)
  );
}

function isFaceAnalysis(result) {
  return !isTongueAnalysis(result);
}

function extractSummaryCards(result) {
  if (isTongueAnalysis(result)) {
    return extractTongueSummaryCards(result);
  }

  const envelope = normalizeAnalysisEnvelope(result);
  const analysisResult = envelope.analysisResult || {};
  const analysis = analysisResult.analysis_results || {};
  const llmReport = analysisResult.llm_report || {};
  const metricReports = llmReport.metric_reports || {};
  const oilSection = analysis.oil_moi || {};
  const skinColorSection = analysis.skin_color || {};
  const sensitivitySection = analysis.sensitivity || {};
  const smoothnessSection = analysis.smoothness || {};
  const wrinklesSection = analysis.wrinkles || {};

  return [
    {
      key: 'oil-moisture',
      reportKey: 'oil_moisture',
      title: '水油状态',
      score: pickNumber(
        metricReports.oil_moisture && metricReports.oil_moisture.score,
        oilSection.score,
        oilSection.oil_analysis && oilSection.oil_analysis.oil_score
      ),
      description: pickText(
        metricReports.oil_moisture && metricReports.oil_moisture.summary,
        oilSection.description,
        oilSection.oil_analysis && oilSection.oil_analysis.description,
        oilSection.moisture_analysis && oilSection.moisture_analysis.description
      ),
      accent: 'blue',
    },
    {
      key: 'sensitivity',
      reportKey: 'sensitivity',
      title: '敏感泛红',
      score: pickNumber(
        metricReports.sensitivity && metricReports.sensitivity.score,
        sensitivitySection.score,
        sensitivitySection.sensitivity_analysis &&
          sensitivitySection.sensitivity_analysis.sensitivity_score
      ),
      description: pickText(
        metricReports.sensitivity && metricReports.sensitivity.summary,
        sensitivitySection.description,
        sensitivitySection.sensitivity_analysis &&
          sensitivitySection.sensitivity_analysis.description
      ),
      accent: 'coral',
    },
    {
      key: 'smoothness',
      reportKey: 'smoothness',
      title: '平滑度',
      score: pickNumber(
        metricReports.smoothness && metricReports.smoothness.score,
        smoothnessSection.score
      ),
      description: pickText(
        metricReports.smoothness && metricReports.smoothness.summary,
        smoothnessSection.description,
        smoothnessSection.smooth &&
          smoothnessSection.smooth.doudou &&
          smoothnessSection.smooth.doudou.suggestion
      ),
      accent: 'green',
    },
    {
      key: 'wrinkles',
      reportKey: 'wrinkles',
      title: '纹理细纹',
      score: pickNumber(
        metricReports.wrinkles && metricReports.wrinkles.score,
        wrinklesSection.black_eye &&
          wrinklesSection.black_eye.position_score &&
          wrinklesSection.black_eye.position_score.score
      ),
      description: pickText(
        metricReports.wrinkles && metricReports.wrinkles.summary,
        wrinklesSection.wrinkles && wrinklesSection.wrinkles.suggest,
        wrinklesSection.black_eye &&
          wrinklesSection.black_eye.suggest &&
          wrinklesSection.black_eye.suggest.talk_suggest
      ),
      accent: 'amber',
    },
    {
      key: 'skin-tone',
      reportKey: 'skin_tone',
      title: '肤色观察',
      score: pickNumber(
        metricReports.skin_tone && metricReports.skin_tone.score,
        skinColorSection.score,
        skinColorSection.hyperpigmentation &&
          skinColorSection.hyperpigmentation.se_ban
      ),
      description: pickText(
        metricReports.skin_tone && metricReports.skin_tone.summary,
        skinColorSection.description,
        skinColorSection.hyperpigmentation &&
          skinColorSection.hyperpigmentation.suggestion,
        skinColorSection.skin_tone_classification &&
          skinColorSection.skin_tone_classification.description
      ),
      accent: 'blue',
    },
  ].map((item) => ({
    ...item,
    scoreText: scoreText(item.score),
  }));
}

function extractOverallReport(result, summaryCards) {
  if (isTongueAnalysis(result)) {
    return extractTongueOverallReport(result, summaryCards);
  }

  const envelope = normalizeAnalysisEnvelope(result);
  const analysisResult = envelope.analysisResult || {};
  const llmReport = analysisResult.llm_report || {};
  const cards = summaryCards || extractSummaryCards(analysisResult);
  const scores = cards
    .map((item) => item.score)
    .filter((value) => typeof value === 'number' && Number.isFinite(value));

  let averageScore = null;
  if (scores.length) {
    averageScore = Math.round(
      scores.reduce((total, value) => total + value, 0) / scores.length
    );
  }

  return {
    totalScore: pickNumber(llmReport.total_score, averageScore),
    totalScoreText: scoreText(pickNumber(llmReport.total_score, averageScore)),
    summary: pickText(
      llmReport.overall_summary,
      '整体结果已生成，建议结合分项评分安排日常护理和复测。'
    ),
    generationMode: llmReport.generation_mode || '',
  };
}

function extractGallery(result) {
  const envelope = normalizeAnalysisEnvelope(result);
  const analysisResult = envelope.analysisResult || {};
  const storage = analysisResult.storage || {};
  const folder = storage.folder || '';
  const uploadedFiles = Array.isArray(analysisResult.uploaded_files)
    ? analysisResult.uploaded_files
    : [];
  const existingImages = Array.isArray(analysisResult.result_images)
    ? analysisResult.result_images
    : [];

  if (existingImages.length) {
    return existingImages;
  }

  if (!folder || !uploadedFiles.length) {
    return [];
  }

  const imageNames = {
    'aligned_face.jpg': '面部对齐图',
    'tongue_segmented.jpg': '舌体分割图',
    'tongue_mask.jpg': '舌体掩膜',
    'tongue_gloss_overlay.jpg': '津润高光叠加',
    'tongue_crack_overlay.jpg': '裂纹候选叠加',
    'tongue_overexposed_overlay.jpg': '过曝区域提示',
    'tongue_moisture_heatmap.jpg': '津润热力图',
    'tongue_moisture_score_breakdown.jpg': '津润评分拆解',
  };

  return uploadedFiles
    .map((item) => {
      const filename = item.filename || '';
      if (!/\.(jpg|jpeg|png|webp)$/i.test(filename)) {
        return null;
      }
      return {
        filename,
        label: imageNames[filename] || filename.replace(/\.[^.]+$/, '').replace(/_/g, ' '),
        object_key: item.object_key || '',
        url: `/v1/mobile/result-image/${folder}/${filename}`,
      };
    })
    .filter(Boolean);
}

function extractTongueSummaryCards(result) {
  const envelope = normalizeAnalysisEnvelope(result);
  const analysis = envelope.analysisResult || {};
  const assistance = analysis.tongue_image_assistance || {};
  const assistanceSummary = assistance.summary || {};
  const moisture = analysis.tongue_moisture || {};
  const coat = analysis.tongue_coat || {};
  const color = analysis.tongue_color || {};
  const crack = analysis.crack_observation || {};
  const quality = assistance.quality || {};

  return [
    {
      key: 'quality',
      title: '图像质量',
      score: null,
      scoreText: quality.level || (quality.passed ? '可分析' : '需重拍'),
      description:
        quality.suggestion || '建议使用光线均匀、舌体完整、画面清晰的照片。',
      accent: quality.passed === false ? 'coral' : 'green',
    },
    {
      key: 'tongue-color',
      title: '舌色倾向',
      score: null,
      scoreText: color.color_name || assistanceSummary.color_tendency || '--',
      description: color.representative_hex
        ? `代表色 ${color.representative_hex}，仅作图像颜色辅助参考。`
        : '未获得稳定舌色特征。',
      accent: 'green',
    },
    {
      key: 'tongue-moisture',
      title: '津润倾向',
      score: typeof moisture.moisture_score === 'number' ? moisture.moisture_score : null,
      scoreText: moisture.moisture_label || scoreText(pickNumber(moisture.moisture_score)),
      description:
        moisture.moisture_explanation ||
        assistanceSummary.moisture_tendency ||
        '暂无津润说明。',
      accent: 'amber',
    },
    {
      key: 'tongue-coat',
      title: '舌苔观察',
      score: typeof coat.coat_coverage_ratio === 'number' ? coat.coat_coverage_ratio : null,
      scoreText: pickRawText(coat.coat_visibility, assistanceSummary.coat_visibility, '--'),
      description: pickRawText(
        coat.coat_color_tendency
          ? `${coat.coat_color_tendency}，${coat.coat_thickness_tendency || '薄厚未定'}`
          : '',
        '当前未见稳定舌苔候选区域。'
      ),
      accent: 'blue',
    },
    {
      key: 'crack',
      title: '裂纹候选',
      score: typeof crack.crack_area_ratio === 'number' ? crack.crack_area_ratio : null,
      scoreText: crack.crack_level || '--',
      description:
        crack.confidence === 'high'
          ? '裂纹候选检测置信度较高，仍需结合人工观察。'
          : '裂纹候选结果仅作参考。',
      accent: 'coral',
    },
  ];
}

function extractTongueOverallReport(result) {
  const envelope = normalizeAnalysisEnvelope(result);
  const analysis = envelope.analysisResult || {};
  const report = analysis.demo_report || {};
  const assistance = analysis.tongue_image_assistance || {};
  const quality = report.quality_gate || assistance.quality || {};
  const tendencies = Array.isArray(report.primary_tendencies)
    ? report.primary_tendencies.join('、')
    : '';

  return {
    totalScore: null,
    totalScoreText: quality.level || (quality.passed ? '可分析' : '需重拍'),
    summary: pickText(
      report.analysis_summary,
      assistance.summary && Array.isArray(assistance.summary.main_observations)
        ? assistance.summary.main_observations.join('；')
        : '',
      tendencies ? `体质倾向参考：${tendencies}` : '',
      '舌象一期辅助分析已完成。'
    ),
    generationMode: report.report_type || 'tongue_phase1_rule_assistance',
  };
}

function extractTongueConstitutionCards(result) {
  const envelope = normalizeAnalysisEnvelope(result);
  const analysis = envelope.analysisResult || {};
  const report = analysis.demo_report || {};
  const tendencies = Array.isArray(report.constitution_tendencies)
    ? report.constitution_tendencies
    : [];
  return tendencies.map((item) => ({
    constitution: item.constitution || '--',
    scoreText: scoreText(pickNumber(item.score)),
    level: item.level || '参考',
    confidence: item.confidence || 'low',
    note: item.note || '仅基于当前舌象图像特征，不作为医学结论。',
    evidence: Array.isArray(item.evidence) ? item.evidence : [],
    missingEvidence: Array.isArray(item.missing_evidence) ? item.missing_evidence : [],
  }));
}

function buildAssistanceContext(result) {
  const envelope = normalizeAnalysisEnvelope(result);
  const analysis = envelope.analysisResult || {};
  const storage = analysis.storage || {};
  const metadata = analysis.metadata || {};
  const summaryCards = extractSummaryCards(analysis);
  const overall = extractOverallReport(analysis, summaryCards);
  const sourceType = isTongueAnalysis(analysis) ? 'tongue' : 'face';

  return {
    sourceType,
    sourceLabel: sourceType === 'tongue' ? '舌象一期' : '面象肤况',
    sourceFolder: storage.folder || metadata.folder || '',
    totalScore: overall.totalScore,
    totalScoreText: overall.totalScoreText,
    summary: overall.summary,
    metricHighlights: summaryCards.map((card) => ({
      key: card.key,
      title: card.title,
      score: card.score,
      summary: card.description,
    })),
    reportUrl: analysis.analysis_report_url || '',
    updatedAt: (metadata && metadata.analysis_timestamp) || '',
  };
}

function sameAssistanceContext(left, right) {
  if (!left || !right) {
    return false;
  }

  const leftFolder = String(left.sourceFolder || left.source_folder || '').trim();
  const rightFolder = String(right.sourceFolder || right.source_folder || '').trim();
  if (leftFolder && rightFolder) {
    return leftFolder === rightFolder;
  }

  return (
    String(left.sourceType || left.source_type || '').trim() ===
      String(right.sourceType || right.source_type || '').trim() &&
    String(left.summary || '').trim() === String(right.summary || '').trim()
  );
}

function contextContainsAssistanceContext(context, target) {
  if (!context || !target) {
    return false;
  }
  if (sameAssistanceContext(context, target)) {
    return true;
  }

  const contexts = Array.isArray(context.diagnosisContexts)
    ? context.diagnosisContexts
    : Array.isArray(context.diagnosis_contexts)
      ? context.diagnosis_contexts
      : [];
  return contexts.some((item) => sameAssistanceContext(item, target));
}

function stringifyPayload(payload) {
  try {
    return JSON.stringify(payload, null, 2);
  } catch (error) {
    return String(payload);
  }
}

module.exports = {
  buildAssistanceContext,
  contextContainsAssistanceContext,
  extractGallery,
  extractOverallReport,
  extractSummaryCards,
  extractTongueConstitutionCards,
  isFaceAnalysis,
  isTongueAnalysis,
  normalizeAnalysisEnvelope,
  sameAssistanceContext,
  stringifyPayload,
};
