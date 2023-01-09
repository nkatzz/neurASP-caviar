
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

	orientation(lb2gt_5,146,1018). coords(lb2gt_5,151,148,1018).  visible(lb2gt_5,1018). happensAt(walking(lb2gt_5),1018). orientation(lb2gt_7,0,1018). coords(lb2gt_7,94,268,1018).  visible(lb2gt_7,1018). happensAt(walking(lb2gt_7),1018).
orientation(lb2gt_5,145,1019). coords(lb2gt_5,152,148,1019).  visible(lb2gt_5,1019). happensAt(walking(lb2gt_5),1019). orientation(lb2gt_7,0,1019). coords(lb2gt_7,94,268,1019).  visible(lb2gt_7,1019). happensAt(walking(lb2gt_7),1019).
orientation(lb2gt_5,146,1020). coords(lb2gt_5,154,150,1020).  visible(lb2gt_5,1020). happensAt(walking(lb2gt_5),1020). orientation(lb2gt_7,0,1020). coords(lb2gt_7,94,268,1020).  visible(lb2gt_7,1020). happensAt(walking(lb2gt_7),1020).
orientation(lb2gt_5,147,1021). coords(lb2gt_5,155,150,1021).  visible(lb2gt_5,1021). happensAt(walking(lb2gt_5),1021). orientation(lb2gt_7,0,1021). coords(lb2gt_7,94,269,1021).  visible(lb2gt_7,1021). happensAt(walking(lb2gt_7),1021).
orientation(lb2gt_5,151,1022). coords(lb2gt_5,156,151,1022).  visible(lb2gt_5,1022). happensAt(walking(lb2gt_5),1022). orientation(lb2gt_7,0,1022). coords(lb2gt_7,94,269,1022).  visible(lb2gt_7,1022). happensAt(walking(lb2gt_7),1022).
orientation(lb2gt_5,152,1023). coords(lb2gt_5,156,151,1023).  visible(lb2gt_5,1023). happensAt(walking(lb2gt_5),1023). orientation(lb2gt_7,0,1023). coords(lb2gt_7,94,269,1023).  visible(lb2gt_7,1023). happensAt(walking(lb2gt_7),1023).
orientation(lb2gt_5,152,1024). coords(lb2gt_5,157,152,1024).  visible(lb2gt_5,1024). happensAt(walking(lb2gt_5),1024). orientation(lb2gt_7,0,1024). coords(lb2gt_7,94,269,1024).  visible(lb2gt_7,1024). happensAt(walking(lb2gt_7),1024).
orientation(lb2gt_5,152,1025). coords(lb2gt_5,157,152,1025).  visible(lb2gt_5,1025). happensAt(walking(lb2gt_5),1025). orientation(lb2gt_7,0,1025). coords(lb2gt_7,93,268,1025).  visible(lb2gt_7,1025). happensAt(walking(lb2gt_7),1025).
orientation(lb2gt_5,152,1026). coords(lb2gt_5,157,152,1026).  visible(lb2gt_5,1026). happensAt(walking(lb2gt_5),1026). orientation(lb2gt_7,0,1026). coords(lb2gt_7,93,268,1026).  visible(lb2gt_7,1026). happensAt(walking(lb2gt_7),1026).
orientation(lb2gt_5,152,1027). coords(lb2gt_5,157,153,1027).  visible(lb2gt_5,1027). happensAt(walking(lb2gt_5),1027). orientation(lb2gt_7,0,1027). coords(lb2gt_7,93,269,1027).  visible(lb2gt_7,1027). happensAt(walking(lb2gt_7),1027).
orientation(lb2gt_5,152,1028). coords(lb2gt_5,159,154,1028).  visible(lb2gt_5,1028). happensAt(walking(lb2gt_5),1028). orientation(lb2gt_7,0,1028). coords(lb2gt_7,93,269,1028).  visible(lb2gt_7,1028). happensAt(walking(lb2gt_7),1028).
orientation(lb2gt_5,152,1029). coords(lb2gt_5,161,156,1029).  visible(lb2gt_5,1029). happensAt(walking(lb2gt_5),1029). orientation(lb2gt_7,0,1029). coords(lb2gt_7,93,268,1029).  visible(lb2gt_7,1029). happensAt(walking(lb2gt_7),1029).
orientation(lb2gt_5,152,1030). coords(lb2gt_5,164,158,1030).  visible(lb2gt_5,1030). happensAt(walking(lb2gt_5),1030). orientation(lb2gt_7,0,1030). coords(lb2gt_7,89,268,1030).  visible(lb2gt_7,1030). happensAt(walking(lb2gt_7),1030).
orientation(lb2gt_5,152,1031). coords(lb2gt_5,166,160,1031).  visible(lb2gt_5,1031). happensAt(walking(lb2gt_5),1031). orientation(lb2gt_7,0,1031). coords(lb2gt_7,89,268,1031).  visible(lb2gt_7,1031). happensAt(walking(lb2gt_7),1031).
orientation(lb2gt_5,152,1032). coords(lb2gt_5,168,161,1032).  visible(lb2gt_5,1032). happensAt(walking(lb2gt_5),1032). orientation(lb2gt_7,0,1032). coords(lb2gt_7,89,268,1032).  visible(lb2gt_7,1032). happensAt(walking(lb2gt_7),1032).
orientation(lb2gt_5,152,1033). coords(lb2gt_5,169,162,1033).  visible(lb2gt_5,1033). happensAt(walking(lb2gt_5),1033). orientation(lb2gt_7,0,1033). coords(lb2gt_7,89,267,1033).  visible(lb2gt_7,1033). happensAt(walking(lb2gt_7),1033).
orientation(lb2gt_5,152,1034). coords(lb2gt_5,170,163,1034).  visible(lb2gt_5,1034). happensAt(walking(lb2gt_5),1034). orientation(lb2gt_7,0,1034). coords(lb2gt_7,88,267,1034).  visible(lb2gt_7,1034). happensAt(walking(lb2gt_7),1034).
orientation(lb2gt_5,152,1035). coords(lb2gt_5,171,164,1035).  visible(lb2gt_5,1035). happensAt(walking(lb2gt_5),1035). orientation(lb2gt_7,0,1035). coords(lb2gt_7,88,267,1035).  visible(lb2gt_7,1035). happensAt(walking(lb2gt_7),1035).
orientation(lb2gt_5,152,1036). coords(lb2gt_5,172,164,1036).  visible(lb2gt_5,1036). happensAt(walking(lb2gt_5),1036). orientation(lb2gt_7,0,1036). coords(lb2gt_7,88,267,1036).  visible(lb2gt_7,1036). happensAt(walking(lb2gt_7),1036).
orientation(lb2gt_5,152,1037). coords(lb2gt_5,172,165,1037).  visible(lb2gt_5,1037). happensAt(walking(lb2gt_5),1037). orientation(lb2gt_7,0,1037). coords(lb2gt_7,88,268,1037).  visible(lb2gt_7,1037). happensAt(walking(lb2gt_7),1037).
orientation(lb2gt_5,152,1038). coords(lb2gt_5,173,166,1038).  visible(lb2gt_5,1038). happensAt(walking(lb2gt_5),1038). orientation(lb2gt_7,0,1038). coords(lb2gt_7,88,267,1038).  visible(lb2gt_7,1038). happensAt(walking(lb2gt_7),1038).
orientation(lb2gt_5,152,1039). coords(lb2gt_5,173,166,1039).  visible(lb2gt_5,1039). happensAt(walking(lb2gt_5),1039). orientation(lb2gt_7,0,1039). coords(lb2gt_7,88,267,1039).  visible(lb2gt_7,1039). happensAt(walking(lb2gt_7),1039).
orientation(lb2gt_5,152,1040). coords(lb2gt_5,173,167,1040).  visible(lb2gt_5,1040). happensAt(walking(lb2gt_5),1040). orientation(lb2gt_7,0,1040). coords(lb2gt_7,88,266,1040).  visible(lb2gt_7,1040). happensAt(walking(lb2gt_7),1040).
orientation(lb2gt_5,152,1041). coords(lb2gt_5,175,168,1041).  visible(lb2gt_5,1041). happensAt(walking(lb2gt_5),1041). orientation(lb2gt_7,0,1041). coords(lb2gt_7,88,266,1041).  visible(lb2gt_7,1041). happensAt(walking(lb2gt_7),1041).
orientation(lb2gt_5,152,1042). coords(lb2gt_5,177,169,1042).  visible(lb2gt_5,1042). happensAt(walking(lb2gt_5),1042). orientation(lb2gt_7,0,1042). coords(lb2gt_7,88,266,1042).  visible(lb2gt_7,1042). happensAt(walking(lb2gt_7),1042).
#show holdsAt/2.