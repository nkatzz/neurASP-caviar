
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

	orientation(wk2gt_1,133,1020). coords(wk2gt_1,136,39,1020).  visible(wk2gt_1,1020). happensAt(walking(wk2gt_1),1020). orientation(wk2gt_2,140,1020). coords(wk2gt_2,97,65,1020).  visible(wk2gt_2,1020). happensAt(inactive(wk2gt_2),1020).
orientation(wk2gt_1,133,1021). coords(wk2gt_1,137,39,1021).  visible(wk2gt_1,1021). happensAt(walking(wk2gt_1),1021). orientation(wk2gt_2,140,1021). coords(wk2gt_2,97,65,1021).  visible(wk2gt_2,1021). happensAt(inactive(wk2gt_2),1021).
orientation(wk2gt_1,133,1022). coords(wk2gt_1,137,39,1022).  visible(wk2gt_1,1022). happensAt(walking(wk2gt_1),1022). orientation(wk2gt_2,140,1022). coords(wk2gt_2,98,65,1022).  visible(wk2gt_2,1022). happensAt(inactive(wk2gt_2),1022).
orientation(wk2gt_1,133,1023). coords(wk2gt_1,137,39,1023).  visible(wk2gt_1,1023). happensAt(walking(wk2gt_1),1023). orientation(wk2gt_2,140,1023). coords(wk2gt_2,98,65,1023).  visible(wk2gt_2,1023). happensAt(inactive(wk2gt_2),1023).
orientation(wk2gt_1,133,1024). coords(wk2gt_1,137,39,1024).  visible(wk2gt_1,1024). happensAt(walking(wk2gt_1),1024). orientation(wk2gt_2,140,1024). coords(wk2gt_2,98,65,1024).  visible(wk2gt_2,1024). happensAt(inactive(wk2gt_2),1024).
orientation(wk2gt_1,133,1025). coords(wk2gt_1,137,39,1025).  visible(wk2gt_1,1025). happensAt(walking(wk2gt_1),1025). orientation(wk2gt_2,140,1025). coords(wk2gt_2,98,65,1025).  visible(wk2gt_2,1025). happensAt(inactive(wk2gt_2),1025).
orientation(wk2gt_1,133,1026). coords(wk2gt_1,137,39,1026).  visible(wk2gt_1,1026). happensAt(walking(wk2gt_1),1026). orientation(wk2gt_2,140,1026). coords(wk2gt_2,98,65,1026).  visible(wk2gt_2,1026). happensAt(inactive(wk2gt_2),1026).
orientation(wk2gt_1,133,1027). coords(wk2gt_1,138,39,1027).  visible(wk2gt_1,1027). happensAt(walking(wk2gt_1),1027). orientation(wk2gt_2,140,1027). coords(wk2gt_2,98,65,1027).  visible(wk2gt_2,1027). happensAt(inactive(wk2gt_2),1027).
orientation(wk2gt_1,133,1028). coords(wk2gt_1,138,38,1028).  visible(wk2gt_1,1028). happensAt(walking(wk2gt_1),1028). orientation(wk2gt_2,140,1028). coords(wk2gt_2,99,64,1028).  visible(wk2gt_2,1028). happensAt(inactive(wk2gt_2),1028).
orientation(wk2gt_1,133,1029). coords(wk2gt_1,138,38,1029).  visible(wk2gt_1,1029). happensAt(walking(wk2gt_1),1029). orientation(wk2gt_2,140,1029). coords(wk2gt_2,99,64,1029).  visible(wk2gt_2,1029). happensAt(inactive(wk2gt_2),1029).
orientation(wk2gt_1,133,1030). coords(wk2gt_1,138,38,1030).  visible(wk2gt_1,1030). happensAt(walking(wk2gt_1),1030). orientation(wk2gt_2,140,1030). coords(wk2gt_2,99,64,1030).  visible(wk2gt_2,1030). happensAt(inactive(wk2gt_2),1030).
orientation(wk2gt_1,133,1031). coords(wk2gt_1,138,38,1031).  visible(wk2gt_1,1031). happensAt(walking(wk2gt_1),1031). orientation(wk2gt_2,140,1031). coords(wk2gt_2,99,64,1031).  visible(wk2gt_2,1031). happensAt(inactive(wk2gt_2),1031).
orientation(wk2gt_1,133,1032). coords(wk2gt_1,139,37,1032).  visible(wk2gt_1,1032). happensAt(walking(wk2gt_1),1032). orientation(wk2gt_2,140,1032). coords(wk2gt_2,99,64,1032).  visible(wk2gt_2,1032). happensAt(inactive(wk2gt_2),1032).
orientation(wk2gt_1,133,1033). coords(wk2gt_1,140,37,1033).  visible(wk2gt_1,1033). happensAt(walking(wk2gt_1),1033). orientation(wk2gt_2,130,1033). coords(wk2gt_2,99,64,1033).  visible(wk2gt_2,1033). happensAt(inactive(wk2gt_2),1033).
orientation(wk2gt_1,133,1034). coords(wk2gt_1,141,37,1034).  visible(wk2gt_1,1034). happensAt(walking(wk2gt_1),1034). orientation(wk2gt_2,129,1034). coords(wk2gt_2,99,64,1034).  visible(wk2gt_2,1034). happensAt(inactive(wk2gt_2),1034).
orientation(wk2gt_1,133,1035). coords(wk2gt_1,141,37,1035).  visible(wk2gt_1,1035). happensAt(walking(wk2gt_1),1035). orientation(wk2gt_2,124,1035). coords(wk2gt_2,99,64,1035).  visible(wk2gt_2,1035). happensAt(active(wk2gt_2),1035).
orientation(wk2gt_1,133,1036). coords(wk2gt_1,141,37,1036).  visible(wk2gt_1,1036). happensAt(walking(wk2gt_1),1036). orientation(wk2gt_2,126,1036). coords(wk2gt_2,99,64,1036).  visible(wk2gt_2,1036). happensAt(active(wk2gt_2),1036).
orientation(wk2gt_1,133,1037). coords(wk2gt_1,141,37,1037).  visible(wk2gt_1,1037). happensAt(walking(wk2gt_1),1037). orientation(wk2gt_2,123,1037). coords(wk2gt_2,99,64,1037).  visible(wk2gt_2,1037). happensAt(active(wk2gt_2),1037).
orientation(wk2gt_1,133,1038). coords(wk2gt_1,142,37,1038).  visible(wk2gt_1,1038). happensAt(walking(wk2gt_1),1038). orientation(wk2gt_2,123,1038). coords(wk2gt_2,99,63,1038).  visible(wk2gt_2,1038). happensAt(active(wk2gt_2),1038).
orientation(wk2gt_1,133,1039). coords(wk2gt_1,142,37,1039).  visible(wk2gt_1,1039). happensAt(walking(wk2gt_1),1039). orientation(wk2gt_2,116,1039). coords(wk2gt_2,99,63,1039).  visible(wk2gt_2,1039). happensAt(active(wk2gt_2),1039).
orientation(wk2gt_1,133,1040). coords(wk2gt_1,142,37,1040).  visible(wk2gt_1,1040). happensAt(walking(wk2gt_1),1040). orientation(wk2gt_2,117,1040). coords(wk2gt_2,99,64,1040).  visible(wk2gt_2,1040). happensAt(active(wk2gt_2),1040).
orientation(wk2gt_1,133,1041). coords(wk2gt_1,143,37,1041).  visible(wk2gt_1,1041). happensAt(walking(wk2gt_1),1041). orientation(wk2gt_2,117,1041). coords(wk2gt_2,99,64,1041).  visible(wk2gt_2,1041). happensAt(active(wk2gt_2),1041).
orientation(wk2gt_1,133,1042). coords(wk2gt_1,143,37,1042).  visible(wk2gt_1,1042). happensAt(walking(wk2gt_1),1042). orientation(wk2gt_2,117,1042). coords(wk2gt_2,99,64,1042).  visible(wk2gt_2,1042). happensAt(active(wk2gt_2),1042).
orientation(wk2gt_1,133,1043). coords(wk2gt_1,144,35,1043).  visible(wk2gt_1,1043). happensAt(walking(wk2gt_1),1043). orientation(wk2gt_2,117,1043). coords(wk2gt_2,99,64,1043).  visible(wk2gt_2,1043). happensAt(active(wk2gt_2),1043).
orientation(wk2gt_1,133,1044). coords(wk2gt_1,144,35,1044).  visible(wk2gt_1,1044). happensAt(walking(wk2gt_1),1044). orientation(wk2gt_2,117,1044). coords(wk2gt_2,98,65,1044).  visible(wk2gt_2,1044). happensAt(active(wk2gt_2),1044).
#show holdsAt/2.