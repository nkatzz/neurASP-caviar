
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(lb1gt_3,134,1029). coords(lb1gt_3,164,69,1029).  visible(lb1gt_3,1029). happensAt(walking(lb1gt_3),1029). orientation(lb1gt_4,90,1029). coords(lb1gt_4,310,112,1029).  visible(lb1gt_4,1029). happensAt(inactive(lb1gt_4),1029).
orientation(lb1gt_3,134,1030). coords(lb1gt_3,163,69,1030).  visible(lb1gt_3,1030). happensAt(walking(lb1gt_3),1030). orientation(lb1gt_4,90,1030). coords(lb1gt_4,310,112,1030).  visible(lb1gt_4,1030). happensAt(inactive(lb1gt_4),1030).
orientation(lb1gt_3,134,1031). coords(lb1gt_3,163,68,1031).  visible(lb1gt_3,1031). happensAt(walking(lb1gt_3),1031). orientation(lb1gt_4,90,1031). coords(lb1gt_4,310,112,1031).  visible(lb1gt_4,1031). happensAt(inactive(lb1gt_4),1031).
orientation(lb1gt_3,134,1032). coords(lb1gt_3,163,67,1032).  visible(lb1gt_3,1032). happensAt(walking(lb1gt_3),1032). orientation(lb1gt_4,90,1032). coords(lb1gt_4,310,112,1032).  visible(lb1gt_4,1032). happensAt(inactive(lb1gt_4),1032).
orientation(lb1gt_3,134,1033). coords(lb1gt_3,162,66,1033).  visible(lb1gt_3,1033). happensAt(walking(lb1gt_3),1033). orientation(lb1gt_4,90,1033). coords(lb1gt_4,310,112,1033).  visible(lb1gt_4,1033). happensAt(inactive(lb1gt_4),1033).
orientation(lb1gt_3,134,1034). coords(lb1gt_3,160,64,1034).  visible(lb1gt_3,1034). happensAt(walking(lb1gt_3),1034). orientation(lb1gt_4,90,1034). coords(lb1gt_4,310,112,1034).  visible(lb1gt_4,1034). happensAt(inactive(lb1gt_4),1034).
orientation(lb1gt_3,134,1035). coords(lb1gt_3,160,62,1035).  visible(lb1gt_3,1035). happensAt(walking(lb1gt_3),1035). orientation(lb1gt_4,90,1035). coords(lb1gt_4,310,112,1035).  visible(lb1gt_4,1035). happensAt(inactive(lb1gt_4),1035).
orientation(lb1gt_3,134,1036). coords(lb1gt_3,158,62,1036).  visible(lb1gt_3,1036). happensAt(walking(lb1gt_3),1036). orientation(lb1gt_4,90,1036). coords(lb1gt_4,310,112,1036).  visible(lb1gt_4,1036). happensAt(inactive(lb1gt_4),1036).
orientation(lb1gt_3,134,1037). coords(lb1gt_3,158,62,1037).  visible(lb1gt_3,1037). happensAt(walking(lb1gt_3),1037). orientation(lb1gt_4,90,1037). coords(lb1gt_4,310,112,1037).  visible(lb1gt_4,1037). happensAt(inactive(lb1gt_4),1037).
orientation(lb1gt_3,134,1038). coords(lb1gt_3,157,61,1038).  visible(lb1gt_3,1038). happensAt(walking(lb1gt_3),1038). orientation(lb1gt_4,90,1038). coords(lb1gt_4,310,112,1038).  visible(lb1gt_4,1038). happensAt(inactive(lb1gt_4),1038).
orientation(lb1gt_3,134,1039). coords(lb1gt_3,156,61,1039).  visible(lb1gt_3,1039). happensAt(walking(lb1gt_3),1039). orientation(lb1gt_4,90,1039). coords(lb1gt_4,310,112,1039).  visible(lb1gt_4,1039). happensAt(inactive(lb1gt_4),1039).
orientation(lb1gt_3,134,1040). coords(lb1gt_3,155,60,1040).  visible(lb1gt_3,1040). happensAt(walking(lb1gt_3),1040). orientation(lb1gt_4,90,1040). coords(lb1gt_4,310,112,1040).  visible(lb1gt_4,1040). happensAt(inactive(lb1gt_4),1040).
orientation(lb1gt_3,134,1041). coords(lb1gt_3,155,60,1041).  visible(lb1gt_3,1041). happensAt(walking(lb1gt_3),1041). orientation(lb1gt_4,90,1041). coords(lb1gt_4,310,112,1041).  visible(lb1gt_4,1041). happensAt(inactive(lb1gt_4),1041).
orientation(lb1gt_3,134,1042). coords(lb1gt_3,156,59,1042).  visible(lb1gt_3,1042). happensAt(walking(lb1gt_3),1042). orientation(lb1gt_4,90,1042). coords(lb1gt_4,310,112,1042).  visible(lb1gt_4,1042). happensAt(inactive(lb1gt_4),1042).
orientation(lb1gt_3,134,1043). coords(lb1gt_3,156,58,1043).  visible(lb1gt_3,1043). happensAt(walking(lb1gt_3),1043). orientation(lb1gt_4,90,1043). coords(lb1gt_4,310,112,1043).  visible(lb1gt_4,1043). happensAt(inactive(lb1gt_4),1043).
orientation(lb1gt_3,134,1044). coords(lb1gt_3,155,57,1044).  visible(lb1gt_3,1044). happensAt(walking(lb1gt_3),1044). orientation(lb1gt_4,90,1044). coords(lb1gt_4,310,112,1044).  visible(lb1gt_4,1044). happensAt(inactive(lb1gt_4),1044).
orientation(lb1gt_3,134,1045). coords(lb1gt_3,155,56,1045).  visible(lb1gt_3,1045). happensAt(walking(lb1gt_3),1045). orientation(lb1gt_4,90,1045). coords(lb1gt_4,310,112,1045).  visible(lb1gt_4,1045). happensAt(inactive(lb1gt_4),1045).
orientation(lb1gt_3,134,1046). coords(lb1gt_3,155,54,1046).  visible(lb1gt_3,1046). happensAt(walking(lb1gt_3),1046). orientation(lb1gt_4,90,1046). coords(lb1gt_4,310,112,1046).  visible(lb1gt_4,1046). happensAt(inactive(lb1gt_4),1046).
orientation(lb1gt_3,134,1047). coords(lb1gt_3,155,52,1047).  visible(lb1gt_3,1047). happensAt(walking(lb1gt_3),1047). orientation(lb1gt_4,90,1047). coords(lb1gt_4,310,112,1047).  visible(lb1gt_4,1047). happensAt(inactive(lb1gt_4),1047).
orientation(lb1gt_3,134,1048). coords(lb1gt_3,155,50,1048).  visible(lb1gt_3,1048). happensAt(walking(lb1gt_3),1048). orientation(lb1gt_4,90,1048). coords(lb1gt_4,310,112,1048).  visible(lb1gt_4,1048). happensAt(inactive(lb1gt_4),1048).
orientation(lb1gt_3,134,1049). coords(lb1gt_3,155,49,1049).  visible(lb1gt_3,1049). happensAt(walking(lb1gt_3),1049). orientation(lb1gt_4,90,1049). coords(lb1gt_4,310,112,1049).  visible(lb1gt_4,1049). happensAt(inactive(lb1gt_4),1049).
orientation(lb1gt_3,134,1050). coords(lb1gt_3,155,48,1050).  visible(lb1gt_3,1050). happensAt(walking(lb1gt_3),1050). orientation(lb1gt_4,90,1050). coords(lb1gt_4,310,112,1050).  visible(lb1gt_4,1050). happensAt(inactive(lb1gt_4),1050).
orientation(lb1gt_3,134,1051). coords(lb1gt_3,155,47,1051).  visible(lb1gt_3,1051). happensAt(walking(lb1gt_3),1051). orientation(lb1gt_4,90,1051). coords(lb1gt_4,310,112,1051).  visible(lb1gt_4,1051). happensAt(inactive(lb1gt_4),1051).
orientation(lb1gt_3,134,1052). coords(lb1gt_3,155,47,1052).  visible(lb1gt_3,1052). happensAt(walking(lb1gt_3),1052). orientation(lb1gt_4,90,1052). coords(lb1gt_4,310,112,1052).  visible(lb1gt_4,1052). happensAt(inactive(lb1gt_4),1052).
orientation(lb1gt_3,134,1053). coords(lb1gt_3,155,46,1053).  visible(lb1gt_3,1053). happensAt(walking(lb1gt_3),1053). orientation(lb1gt_4,90,1053). coords(lb1gt_4,310,112,1053).  visible(lb1gt_4,1053). happensAt(inactive(lb1gt_4),1053).
#show holdsAt/2.