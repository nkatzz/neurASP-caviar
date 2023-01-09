
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

	orientation(br1gt_2,174,1018). coords(br1gt_2,111,214,1018).  visible(br1gt_2,1018). happensAt(walking(br1gt_2),1018). orientation(br1gt_3,135,1018). coords(br1gt_3,233,95,1018).  visible(br1gt_3,1018). happensAt(walking(br1gt_3),1018).
orientation(br1gt_2,174,1019). coords(br1gt_2,111,215,1019).  visible(br1gt_2,1019). happensAt(walking(br1gt_2),1019). orientation(br1gt_3,135,1019). coords(br1gt_3,233,95,1019).  visible(br1gt_3,1019). happensAt(walking(br1gt_3),1019).
orientation(br1gt_2,174,1020). coords(br1gt_2,111,215,1020).  visible(br1gt_2,1020). happensAt(active(br1gt_2),1020). orientation(br1gt_3,135,1020). coords(br1gt_3,231,95,1020).  visible(br1gt_3,1020). happensAt(walking(br1gt_3),1020).
orientation(br1gt_2,174,1021). coords(br1gt_2,111,215,1021).  visible(br1gt_2,1021). happensAt(active(br1gt_2),1021). orientation(br1gt_3,135,1021). coords(br1gt_3,231,96,1021).  visible(br1gt_3,1021). happensAt(walking(br1gt_3),1021).
orientation(br1gt_2,174,1022). coords(br1gt_2,111,216,1022).  visible(br1gt_2,1022). happensAt(active(br1gt_2),1022). orientation(br1gt_3,135,1022). coords(br1gt_3,230,96,1022).  visible(br1gt_3,1022). happensAt(walking(br1gt_3),1022).
orientation(br1gt_2,174,1023). coords(br1gt_2,111,216,1023).  visible(br1gt_2,1023). happensAt(active(br1gt_2),1023). orientation(br1gt_3,135,1023). coords(br1gt_3,229,96,1023).  visible(br1gt_3,1023). happensAt(walking(br1gt_3),1023).
orientation(br1gt_2,174,1024). coords(br1gt_2,111,216,1024).  visible(br1gt_2,1024). happensAt(active(br1gt_2),1024). orientation(br1gt_3,135,1024). coords(br1gt_3,228,96,1024).  visible(br1gt_3,1024). happensAt(walking(br1gt_3),1024).
orientation(br1gt_2,174,1025). coords(br1gt_2,111,216,1025).  visible(br1gt_2,1025). happensAt(active(br1gt_2),1025). orientation(br1gt_3,137,1025). coords(br1gt_3,226,96,1025).  visible(br1gt_3,1025). happensAt(walking(br1gt_3),1025).
orientation(br1gt_2,174,1026). coords(br1gt_2,111,216,1026).  visible(br1gt_2,1026). happensAt(active(br1gt_2),1026). orientation(br1gt_3,137,1026). coords(br1gt_3,224,96,1026).  visible(br1gt_3,1026). happensAt(walking(br1gt_3),1026).
orientation(br1gt_2,174,1027). coords(br1gt_2,111,216,1027).  visible(br1gt_2,1027). happensAt(active(br1gt_2),1027). orientation(br1gt_3,137,1027). coords(br1gt_3,221,96,1027).  visible(br1gt_3,1027). happensAt(walking(br1gt_3),1027).
orientation(br1gt_2,174,1028). coords(br1gt_2,111,216,1028).  visible(br1gt_2,1028). happensAt(active(br1gt_2),1028). orientation(br1gt_3,137,1028). coords(br1gt_3,219,96,1028).  visible(br1gt_3,1028). happensAt(walking(br1gt_3),1028).
orientation(br1gt_2,174,1029). coords(br1gt_2,111,216,1029).  visible(br1gt_2,1029). happensAt(active(br1gt_2),1029). orientation(br1gt_3,137,1029). coords(br1gt_3,217,97,1029).  visible(br1gt_3,1029). happensAt(walking(br1gt_3),1029).
orientation(br1gt_2,174,1030). coords(br1gt_2,111,219,1030).  visible(br1gt_2,1030). happensAt(active(br1gt_2),1030). orientation(br1gt_3,137,1030). coords(br1gt_3,214,97,1030).  visible(br1gt_3,1030). happensAt(walking(br1gt_3),1030).
orientation(br1gt_2,174,1031). coords(br1gt_2,111,219,1031).  visible(br1gt_2,1031). happensAt(active(br1gt_2),1031). orientation(br1gt_3,137,1031). coords(br1gt_3,212,97,1031).  visible(br1gt_3,1031). happensAt(walking(br1gt_3),1031).
orientation(br1gt_2,174,1032). coords(br1gt_2,111,219,1032).  visible(br1gt_2,1032). happensAt(active(br1gt_2),1032). orientation(br1gt_3,137,1032). coords(br1gt_3,210,97,1032).  visible(br1gt_3,1032). happensAt(walking(br1gt_3),1032).
orientation(br1gt_2,174,1033). coords(br1gt_2,111,219,1033).  visible(br1gt_2,1033). happensAt(active(br1gt_2),1033). orientation(br1gt_3,137,1033). coords(br1gt_3,210,97,1033).  visible(br1gt_3,1033). happensAt(walking(br1gt_3),1033).
orientation(br1gt_2,174,1034). coords(br1gt_2,113,218,1034).  visible(br1gt_2,1034). happensAt(active(br1gt_2),1034). orientation(br1gt_3,137,1034). coords(br1gt_3,209,97,1034).  visible(br1gt_3,1034). happensAt(walking(br1gt_3),1034).
orientation(br1gt_2,174,1035). coords(br1gt_2,113,218,1035).  visible(br1gt_2,1035). happensAt(active(br1gt_2),1035). orientation(br1gt_3,137,1035). coords(br1gt_3,209,97,1035).  visible(br1gt_3,1035). happensAt(walking(br1gt_3),1035).
orientation(br1gt_2,174,1036). coords(br1gt_2,111,218,1036).  visible(br1gt_2,1036). happensAt(active(br1gt_2),1036). orientation(br1gt_3,137,1036). coords(br1gt_3,208,97,1036).  visible(br1gt_3,1036). happensAt(walking(br1gt_3),1036).
orientation(br1gt_2,174,1037). coords(br1gt_2,113,218,1037).  visible(br1gt_2,1037). happensAt(active(br1gt_2),1037). orientation(br1gt_3,137,1037). coords(br1gt_3,207,97,1037).  visible(br1gt_3,1037). happensAt(walking(br1gt_3),1037).
orientation(br1gt_2,174,1038). coords(br1gt_2,113,218,1038).  visible(br1gt_2,1038). happensAt(active(br1gt_2),1038). orientation(br1gt_3,137,1038). coords(br1gt_3,207,97,1038).  visible(br1gt_3,1038). happensAt(walking(br1gt_3),1038).
orientation(br1gt_2,174,1039). coords(br1gt_2,113,218,1039).  visible(br1gt_2,1039). happensAt(active(br1gt_2),1039). orientation(br1gt_3,137,1039). coords(br1gt_3,206,97,1039).  visible(br1gt_3,1039). happensAt(walking(br1gt_3),1039).
orientation(br1gt_2,169,1040). coords(br1gt_2,113,217,1040).  visible(br1gt_2,1040). happensAt(active(br1gt_2),1040). orientation(br1gt_3,138,1040). coords(br1gt_3,205,97,1040).  visible(br1gt_3,1040). happensAt(walking(br1gt_3),1040).
orientation(br1gt_2,169,1041). coords(br1gt_2,113,217,1041).  visible(br1gt_2,1041). happensAt(active(br1gt_2),1041). orientation(br1gt_3,141,1041). coords(br1gt_3,205,97,1041).  visible(br1gt_3,1041). happensAt(walking(br1gt_3),1041).
orientation(br1gt_2,169,1042). coords(br1gt_2,114,217,1042).  happensAt(disappear(br1gt_2),1042). happensAt(active(br1gt_2),1042). orientation(br1gt_3,141,1042). coords(br1gt_3,203,97,1042).  happensAt(disappear(br1gt_3),1042). happensAt(walking(br1gt_3),1042).
#show holdsAt/2.