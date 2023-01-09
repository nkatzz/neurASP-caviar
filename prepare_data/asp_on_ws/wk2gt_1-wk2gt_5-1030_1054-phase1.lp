
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

	orientation(wk2gt_1,133,1030). coords(wk2gt_1,138,38,1030).  visible(wk2gt_1,1030). happensAt(walking(wk2gt_1),1030). orientation(wk2gt_5,150,1030). coords(wk2gt_5,137,128,1030).  visible(wk2gt_5,1030). happensAt(walking(wk2gt_5),1030).
orientation(wk2gt_1,133,1031). coords(wk2gt_1,138,38,1031).  visible(wk2gt_1,1031). happensAt(walking(wk2gt_1),1031). orientation(wk2gt_5,150,1031). coords(wk2gt_5,137,128,1031).  visible(wk2gt_5,1031). happensAt(walking(wk2gt_5),1031).
orientation(wk2gt_1,133,1032). coords(wk2gt_1,139,37,1032).  visible(wk2gt_1,1032). happensAt(walking(wk2gt_1),1032). orientation(wk2gt_5,150,1032). coords(wk2gt_5,137,128,1032).  visible(wk2gt_5,1032). happensAt(walking(wk2gt_5),1032).
orientation(wk2gt_1,133,1033). coords(wk2gt_1,140,37,1033).  visible(wk2gt_1,1033). happensAt(walking(wk2gt_1),1033). orientation(wk2gt_5,150,1033). coords(wk2gt_5,138,129,1033).  visible(wk2gt_5,1033). happensAt(walking(wk2gt_5),1033).
orientation(wk2gt_1,133,1034). coords(wk2gt_1,141,37,1034).  visible(wk2gt_1,1034). happensAt(walking(wk2gt_1),1034). orientation(wk2gt_5,150,1034). coords(wk2gt_5,138,130,1034).  visible(wk2gt_5,1034). happensAt(walking(wk2gt_5),1034).
orientation(wk2gt_1,133,1035). coords(wk2gt_1,141,37,1035).  visible(wk2gt_1,1035). happensAt(walking(wk2gt_1),1035). orientation(wk2gt_5,150,1035). coords(wk2gt_5,138,132,1035).  visible(wk2gt_5,1035). happensAt(walking(wk2gt_5),1035).
orientation(wk2gt_1,133,1036). coords(wk2gt_1,141,37,1036).  visible(wk2gt_1,1036). happensAt(walking(wk2gt_1),1036). orientation(wk2gt_5,150,1036). coords(wk2gt_5,139,133,1036).  visible(wk2gt_5,1036). happensAt(walking(wk2gt_5),1036).
orientation(wk2gt_1,133,1037). coords(wk2gt_1,141,37,1037).  visible(wk2gt_1,1037). happensAt(walking(wk2gt_1),1037). orientation(wk2gt_5,150,1037). coords(wk2gt_5,140,135,1037).  visible(wk2gt_5,1037). happensAt(walking(wk2gt_5),1037).
orientation(wk2gt_1,133,1038). coords(wk2gt_1,142,37,1038).  visible(wk2gt_1,1038). happensAt(walking(wk2gt_1),1038). orientation(wk2gt_5,150,1038). coords(wk2gt_5,143,136,1038).  visible(wk2gt_5,1038). happensAt(walking(wk2gt_5),1038).
orientation(wk2gt_1,133,1039). coords(wk2gt_1,142,37,1039).  visible(wk2gt_1,1039). happensAt(walking(wk2gt_1),1039). orientation(wk2gt_5,150,1039). coords(wk2gt_5,144,138,1039).  visible(wk2gt_5,1039). happensAt(walking(wk2gt_5),1039).
orientation(wk2gt_1,133,1040). coords(wk2gt_1,142,37,1040).  visible(wk2gt_1,1040). happensAt(walking(wk2gt_1),1040). orientation(wk2gt_5,150,1040). coords(wk2gt_5,146,139,1040).  visible(wk2gt_5,1040). happensAt(walking(wk2gt_5),1040).
orientation(wk2gt_1,133,1041). coords(wk2gt_1,143,37,1041).  visible(wk2gt_1,1041). happensAt(walking(wk2gt_1),1041). orientation(wk2gt_5,150,1041). coords(wk2gt_5,146,140,1041).  visible(wk2gt_5,1041). happensAt(walking(wk2gt_5),1041).
orientation(wk2gt_1,133,1042). coords(wk2gt_1,143,37,1042).  visible(wk2gt_1,1042). happensAt(walking(wk2gt_1),1042). orientation(wk2gt_5,150,1042). coords(wk2gt_5,147,141,1042).  visible(wk2gt_5,1042). happensAt(walking(wk2gt_5),1042).
orientation(wk2gt_1,133,1043). coords(wk2gt_1,144,35,1043).  visible(wk2gt_1,1043). happensAt(walking(wk2gt_1),1043). orientation(wk2gt_5,150,1043). coords(wk2gt_5,147,141,1043).  visible(wk2gt_5,1043). happensAt(walking(wk2gt_5),1043).
orientation(wk2gt_1,133,1044). coords(wk2gt_1,144,35,1044).  visible(wk2gt_1,1044). happensAt(walking(wk2gt_1),1044). orientation(wk2gt_5,150,1044). coords(wk2gt_5,148,142,1044).  visible(wk2gt_5,1044). happensAt(walking(wk2gt_5),1044).
orientation(wk2gt_1,133,1045). coords(wk2gt_1,144,35,1045).  visible(wk2gt_1,1045). happensAt(walking(wk2gt_1),1045). orientation(wk2gt_5,150,1045). coords(wk2gt_5,148,142,1045).  visible(wk2gt_5,1045). happensAt(walking(wk2gt_5),1045).
orientation(wk2gt_1,133,1046). coords(wk2gt_1,144,35,1046).  visible(wk2gt_1,1046). happensAt(walking(wk2gt_1),1046). orientation(wk2gt_5,150,1046). coords(wk2gt_5,148,143,1046).  visible(wk2gt_5,1046). happensAt(walking(wk2gt_5),1046).
orientation(wk2gt_1,133,1047). coords(wk2gt_1,144,35,1047).  visible(wk2gt_1,1047). happensAt(walking(wk2gt_1),1047). orientation(wk2gt_5,150,1047). coords(wk2gt_5,150,143,1047).  visible(wk2gt_5,1047). happensAt(walking(wk2gt_5),1047).
orientation(wk2gt_1,133,1048). coords(wk2gt_1,144,35,1048).  visible(wk2gt_1,1048). happensAt(walking(wk2gt_1),1048). orientation(wk2gt_5,150,1048). coords(wk2gt_5,152,144,1048).  visible(wk2gt_5,1048). happensAt(walking(wk2gt_5),1048).
orientation(wk2gt_1,133,1049). coords(wk2gt_1,145,35,1049).  visible(wk2gt_1,1049). happensAt(walking(wk2gt_1),1049). orientation(wk2gt_5,155,1049). coords(wk2gt_5,154,145,1049).  visible(wk2gt_5,1049). happensAt(walking(wk2gt_5),1049).
orientation(wk2gt_1,133,1050). coords(wk2gt_1,145,35,1050).  visible(wk2gt_1,1050). happensAt(walking(wk2gt_1),1050). orientation(wk2gt_5,155,1050). coords(wk2gt_5,156,147,1050).  visible(wk2gt_5,1050). happensAt(walking(wk2gt_5),1050).
orientation(wk2gt_1,133,1051). coords(wk2gt_1,145,35,1051).  visible(wk2gt_1,1051). happensAt(walking(wk2gt_1),1051). orientation(wk2gt_5,155,1051). coords(wk2gt_5,158,149,1051).  visible(wk2gt_5,1051). happensAt(walking(wk2gt_5),1051).
orientation(wk2gt_1,133,1052). coords(wk2gt_1,146,34,1052).  visible(wk2gt_1,1052). happensAt(walking(wk2gt_1),1052). orientation(wk2gt_5,155,1052). coords(wk2gt_5,159,151,1052).  visible(wk2gt_5,1052). happensAt(walking(wk2gt_5),1052).
orientation(wk2gt_1,133,1053). coords(wk2gt_1,145,34,1053).  visible(wk2gt_1,1053). happensAt(walking(wk2gt_1),1053). orientation(wk2gt_5,155,1053). coords(wk2gt_5,161,153,1053).  visible(wk2gt_5,1053). happensAt(walking(wk2gt_5),1053).
orientation(wk2gt_1,133,1054). coords(wk2gt_1,145,34,1054).  happensAt(disappear(wk2gt_1),1054). happensAt(walking(wk2gt_1),1054). orientation(wk2gt_5,155,1054). coords(wk2gt_5,161,154,1054).  happensAt(disappear(wk2gt_5),1054). happensAt(walking(wk2gt_5),1054).
#show holdsAt/2.