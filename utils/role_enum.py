from enum import Enum
from typing import Dict


class RoleEnum(str, Enum):
    PATIENT_DENTAL = "patient_dental"
    PATIENT_DIABETES = "patient_diabetes"
    DOCTOR_DENTAL = "doctor_dental"
    DOCTOR_ENDOCRINE = "doctor_endocrine"


ROLE_DISPLAY_NAME: Dict[RoleEnum, str] = {
    RoleEnum.PATIENT_DENTAL: "Bệnh nhân nha khoa",
    RoleEnum.PATIENT_DIABETES: "Bệnh nhân đái tháo đường",
    RoleEnum.DOCTOR_DENTAL: "Bác sĩ nha khoa",
    RoleEnum.DOCTOR_ENDOCRINE: "Bác sĩ nội tiết",
    RoleEnum.ORTHODONTIST: "Bác sĩ chỉnh nha",
}
ROLE_DESCRIPTION_BY_VALUE = {
    RoleEnum.PATIENT_DENTAL.value: "Dành cho người cần tư vấn về các vấn đề răng miệng, nha chu, và chăm sóc sức khỏe răng miệng",
    RoleEnum.PATIENT_DIABETES.value: "Dành cho người mắc đái tháo đường cần tư vấn về mối liên hệ giữa bệnh đái tháo đường và sức khỏe răng miệng",
    RoleEnum.DOCTOR_DENTAL.value: "Dành cho bác sĩ nha khoa cần tư vấn về tác động của đái tháo đường đến điều trị nha khoa",
    RoleEnum.DOCTOR_ENDOCRINE.value: "Dành cho bác sĩ nội tiết cần hiểu về biến chứng răng miệng ở bệnh nhân đái tháo đường",
    RoleEnum.ORTHODONTIST.value: "Dành cho bác sĩ chỉnh nha cần tham khảo kiến thức y khoa liên quan nha khoa",
}
ROLE_DESCRIPTION: Dict[RoleEnum, str] = {
    RoleEnum.PATIENT_DENTAL : "Dành cho người cần tư vấn về các vấn đề răng miệng, nha chu, và chăm sóc sức khỏe răng miệng",
    RoleEnum.PATIENT_DIABETES: "Dành cho người mắc đái tháo đường cần tư vấn về mối liên hệ giữa bệnh đái tháo đường và sức khỏe răng miệng",
    RoleEnum.DOCTOR_DENTAL: "Dành cho bác sĩ nha khoa cần tư vấn về tác động của đái tháo đường đến điều trị nha khoa",
    RoleEnum.DOCTOR_ENDOCRINE: "Dành cho bác sĩ nội tiết cần hiểu về biến chứng răng miệng ở bệnh nhân đái tháo đường",
    RoleEnum.ORTHODONTIST: "Dành cho bác sĩ chỉnh nha cần tham khảo kiến thức y khoa liên quan nha khoa",
}
PERSONA_BY_ROLE: Dict[str, Dict[str, str]] = {
    RoleEnum.DOCTOR_DENTAL.value: {
        "persona": "Bác sĩ nội tiết (chuyên ĐTĐ)",
        "audience": "bác sĩ nha khoa",
        "tone": (
            "Thái độ thân thiện, giải thích ngắn gọn, dùng từ phù hợp để bác sĩ nha khoa hiểu, mục tiêu giúp bác sĩ nha khoa hiểu hơn về nội tiết "
        ),
    },
    RoleEnum.DOCTOR_ENDOCRINE.value: {
        "persona": "Bác sĩ nha khoa",
        "audience": "bác sĩ nội tiết",
        "tone": (
            "Thái độ thân thiện, giải thích ngắn gọn, dùng từ phù hợp để bác sĩ nội tiết hiểu, mục tiêu giúp bác sĩ nội tiết hiểu hơn về nha khoa phục vụ cho nghành nghề của mình"
        ),
    },
    RoleEnum.PATIENT_DIABETES.value: {
        "persona": "Bác sĩ nội tiết",
        "audience": "bệnh nhân đái tháo đường",
        "tone": (
            "Thái độ thân thiện, Ngôn ngữ giản dị, không nói dài dòng,không dùng từ chuyên môn. "
        ),
    },
    RoleEnum.PATIENT_DENTAL.value: {
        "persona": "Bác sĩ nha khoa",
        "audience": "bệnh nhân nha khoa",
        "tone": (
            "Thái độ thân thiện, Ngôn ngữ thân thiện, không  dùng từ chuyên môn"
        ),
    },
    RoleEnum.ORTHODONTIST.value: {
        "persona": "Bác sĩ chỉnh nha",
        "audience": "bác sĩ nha khoa (chỉnh nha)",
        "tone": (
            "Chính xác, súc tích, tập trung bằng chứng; tránh dài dòng; phù hợp đồng nghiệp chuyên môn"
        ),
    },
}
