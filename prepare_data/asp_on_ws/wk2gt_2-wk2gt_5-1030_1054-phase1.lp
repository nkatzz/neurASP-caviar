
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

	orientation(wk2gt_2,140,1030). coords(wk2gt_2,99,64,1030).  visible(wk2gt_2,1030). happensAt(inactive(wk2gt_2),1030). orientation(wk2gt_5,150,1030). coords(wk2gt_5,137,128,1030).  visible(wk2gt_5,1030). happensAt(walking(wk2gt_5),1030).
orientation(wk2gt_2,140,1031). coords(wk2gt_2,99,64,1031).  visible(wk2gt_2,1031). happensAt(inactive(wk2gt_2),1031). orientation(wk2gt_5,150,1031). coords(wk2gt_5,137,128,1031).  visible(wk2gt_5,1031). happensAt(walking(wk2gt_5),1031).
orientation(wk2gt_2,140,1032). coords(wk2gt_2,99,64,1032).  visible(wk2gt_2,1032). happensAt(inactive(wk2gt_2),1032). orientation(wk2gt_5,150,1032). coords(wk2gt_5,137,128,1032).  visible(wk2gt_5,1032). happensAt(walking(wk2gt_5),1032).
orientation(wk2gt_2,130,1033). coords(wk2gt_2,99,64,1033).  visible(wk2gt_2,1033). happensAt(inactive(wk2gt_2),1033). orientation(wk2gt_5,150,1033). coords(wk2gt_5,138,129,1033).  visible(wk2gt_5,1033). happensAt(walking(wk2gt_5),1033).
orientation(wk2gt_2,129,1034). coords(wk2gt_2,99,64,1034).  visible(wk2gt_2,1034). happensAt(inactive(wk2gt_2),1034). orientation(wk2gt_5,150,1034). coords(wk2gt_5,138,130,1034).  visible(wk2gt_5,1034). happensAt(walking(wk2gt_5),1034).
orientation(wk2gt_2,124,1035). coords(wk2gt_2,99,64,1035).  visible(wk2gt_2,1035). happensAt(active(wk2gt_2),1035). orientation(wk2gt_5,150,1035). coords(wk2gt_5,138,132,1035).  visible(wk2gt_5,1035). happensAt(walking(wk2gt_5),1035).
orientation(wk2gt_2,126,1036). coords(wk2gt_2,99,64,1036).  visible(wk2gt_2,1036). happensAt(active(wk2gt_2),1036). orientation(wk2gt_5,150,1036). coords(wk2gt_5,139,133,1036).  visible(wk2gt_5,1036). happensAt(walking(wk2gt_5),1036).
orientation(wk2gt_2,123,1037). coords(wk2gt_2,99,64,1037).  visible(wk2gt_2,1037). happensAt(active(wk2gt_2),1037). orientation(wk2gt_5,150,1037). coords(wk2gt_5,140,135,1037).  visible(wk2gt_5,1037). happensAt(walking(wk2gt_5),1037).
orientation(wk2gt_2,123,1038). coords(wk2gt_2,99,63,1038).  visible(wk2gt_2,1038). happensAt(active(wk2gt_2),1038). orientation(wk2gt_5,150,1038). coords(wk2gt_5,143,136,1038).  visible(wk2gt_5,1038). happensAt(walking(wk2gt_5),1038).
orientation(wk2gt_2,116,1039). coords(wk2gt_2,99,63,1039).  visible(wk2gt_2,1039). happensAt(active(wk2gt_2),1039). orientation(wk2gt_5,150,1039). coords(wk2gt_5,144,138,1039).  visible(wk2gt_5,1039). happensAt(walking(wk2gt_5),1039).
orientation(wk2gt_2,117,1040). coords(wk2gt_2,99,64,1040).  visible(wk2gt_2,1040). happensAt(active(wk2gt_2),1040). orientation(wk2gt_5,150,1040). coords(wk2gt_5,146,139,1040).  visible(wk2gt_5,1040). happensAt(walking(wk2gt_5),1040).
orientation(wk2gt_2,117,1041). coords(wk2gt_2,99,64,1041).  visible(wk2gt_2,1041). happensAt(active(wk2gt_2),1041). orientation(wk2gt_5,150,1041). coords(wk2gt_5,146,140,1041).  visible(wk2gt_5,1041). happensAt(walking(wk2gt_5),1041).
orientation(wk2gt_2,117,1042). coords(wk2gt_2,99,64,1042).  visible(wk2gt_2,1042). happensAt(active(wk2gt_2),1042). orientation(wk2gt_5,150,1042). coords(wk2gt_5,147,141,1042).  visible(wk2gt_5,1042). happensAt(walking(wk2gt_5),1042).
orientation(wk2gt_2,117,1043). coords(wk2gt_2,99,64,1043).  visible(wk2gt_2,1043). happensAt(active(wk2gt_2),1043). orientation(wk2gt_5,150,1043). coords(wk2gt_5,147,141,1043).  visible(wk2gt_5,1043). happensAt(walking(wk2gt_5),1043).
orientation(wk2gt_2,117,1044). coords(wk2gt_2,98,65,1044).  visible(wk2gt_2,1044). happensAt(active(wk2gt_2),1044). orientation(wk2gt_5,150,1044). coords(wk2gt_5,148,142,1044).  visible(wk2gt_5,1044). happensAt(walking(wk2gt_5),1044).
orientation(wk2gt_2,117,1045). coords(wk2gt_2,98,65,1045).  visible(wk2gt_2,1045). happensAt(active(wk2gt_2),1045). orientation(wk2gt_5,150,1045). coords(wk2gt_5,148,142,1045).  visible(wk2gt_5,1045). happensAt(walking(wk2gt_5),1045).
orientation(wk2gt_2,117,1046). coords(wk2gt_2,99,65,1046).  visible(wk2gt_2,1046). happensAt(active(wk2gt_2),1046). orientation(wk2gt_5,150,1046). coords(wk2gt_5,148,143,1046).  visible(wk2gt_5,1046). happensAt(walking(wk2gt_5),1046).
orientation(wk2gt_2,117,1047). coords(wk2gt_2,98,65,1047).  visible(wk2gt_2,1047). happensAt(active(wk2gt_2),1047). orientation(wk2gt_5,150,1047). coords(wk2gt_5,150,143,1047).  visible(wk2gt_5,1047). happensAt(walking(wk2gt_5),1047).
orientation(wk2gt_2,117,1048). coords(wk2gt_2,98,65,1048).  visible(wk2gt_2,1048). happensAt(active(wk2gt_2),1048). orientation(wk2gt_5,150,1048). coords(wk2gt_5,152,144,1048).  visible(wk2gt_5,1048). happensAt(walking(wk2gt_5),1048).
orientation(wk2gt_2,117,1049). coords(wk2gt_2,98,66,1049).  visible(wk2gt_2,1049). happensAt(active(wk2gt_2),1049). orientation(wk2gt_5,155,1049). coords(wk2gt_5,154,145,1049).  visible(wk2gt_5,1049). happensAt(walking(wk2gt_5),1049).
orientation(wk2gt_2,117,1050). coords(wk2gt_2,98,66,1050).  visible(wk2gt_2,1050). happensAt(active(wk2gt_2),1050). orientation(wk2gt_5,155,1050). coords(wk2gt_5,156,147,1050).  visible(wk2gt_5,1050). happensAt(walking(wk2gt_5),1050).
orientation(wk2gt_2,117,1051). coords(wk2gt_2,98,66,1051).  visible(wk2gt_2,1051). happensAt(active(wk2gt_2),1051). orientation(wk2gt_5,155,1051). coords(wk2gt_5,158,149,1051).  visible(wk2gt_5,1051). happensAt(walking(wk2gt_5),1051).
orientation(wk2gt_2,117,1052). coords(wk2gt_2,98,66,1052).  visible(wk2gt_2,1052). happensAt(active(wk2gt_2),1052). orientation(wk2gt_5,155,1052). coords(wk2gt_5,159,151,1052).  visible(wk2gt_5,1052). happensAt(walking(wk2gt_5),1052).
orientation(wk2gt_2,117,1053). coords(wk2gt_2,97,67,1053).  visible(wk2gt_2,1053). happensAt(active(wk2gt_2),1053). orientation(wk2gt_5,155,1053). coords(wk2gt_5,161,153,1053).  visible(wk2gt_5,1053). happensAt(walking(wk2gt_5),1053).
orientation(wk2gt_2,117,1054). coords(wk2gt_2,97,67,1054).  happensAt(disappear(wk2gt_2),1054). happensAt(active(wk2gt_2),1054). orientation(wk2gt_5,155,1054). coords(wk2gt_5,161,154,1054).  happensAt(disappear(wk2gt_5),1054). happensAt(walking(wk2gt_5),1054).
#show holdsAt/2.