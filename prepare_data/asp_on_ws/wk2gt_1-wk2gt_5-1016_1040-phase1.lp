
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

	orientation(wk2gt_1,133,1016). coords(wk2gt_1,136,39,1016).  visible(wk2gt_1,1016). happensAt(walking(wk2gt_1),1016). orientation(wk2gt_5,150,1016). coords(wk2gt_5,122,124,1016).  visible(wk2gt_5,1016). happensAt(walking(wk2gt_5),1016).
orientation(wk2gt_1,133,1017). coords(wk2gt_1,136,39,1017).  visible(wk2gt_1,1017). happensAt(walking(wk2gt_1),1017). orientation(wk2gt_5,150,1017). coords(wk2gt_5,122,125,1017).  visible(wk2gt_5,1017). happensAt(walking(wk2gt_5),1017).
orientation(wk2gt_1,133,1018). coords(wk2gt_1,136,39,1018).  visible(wk2gt_1,1018). happensAt(walking(wk2gt_1),1018). orientation(wk2gt_5,150,1018). coords(wk2gt_5,123,125,1018).  visible(wk2gt_5,1018). happensAt(walking(wk2gt_5),1018).
orientation(wk2gt_1,133,1019). coords(wk2gt_1,136,39,1019).  visible(wk2gt_1,1019). happensAt(walking(wk2gt_1),1019). orientation(wk2gt_5,150,1019). coords(wk2gt_5,123,125,1019).  visible(wk2gt_5,1019). happensAt(walking(wk2gt_5),1019).
orientation(wk2gt_1,133,1020). coords(wk2gt_1,136,39,1020).  visible(wk2gt_1,1020). happensAt(walking(wk2gt_1),1020). orientation(wk2gt_5,150,1020). coords(wk2gt_5,123,125,1020).  visible(wk2gt_5,1020). happensAt(walking(wk2gt_5),1020).
orientation(wk2gt_1,133,1021). coords(wk2gt_1,137,39,1021).  visible(wk2gt_1,1021). happensAt(walking(wk2gt_1),1021). orientation(wk2gt_5,150,1021). coords(wk2gt_5,123,125,1021).  visible(wk2gt_5,1021). happensAt(walking(wk2gt_5),1021).
orientation(wk2gt_1,133,1022). coords(wk2gt_1,137,39,1022).  visible(wk2gt_1,1022). happensAt(walking(wk2gt_1),1022). orientation(wk2gt_5,150,1022). coords(wk2gt_5,124,125,1022).  visible(wk2gt_5,1022). happensAt(walking(wk2gt_5),1022).
orientation(wk2gt_1,133,1023). coords(wk2gt_1,137,39,1023).  visible(wk2gt_1,1023). happensAt(walking(wk2gt_1),1023). orientation(wk2gt_5,150,1023). coords(wk2gt_5,127,125,1023).  visible(wk2gt_5,1023). happensAt(walking(wk2gt_5),1023).
orientation(wk2gt_1,133,1024). coords(wk2gt_1,137,39,1024).  visible(wk2gt_1,1024). happensAt(walking(wk2gt_1),1024). orientation(wk2gt_5,150,1024). coords(wk2gt_5,129,125,1024).  visible(wk2gt_5,1024). happensAt(walking(wk2gt_5),1024).
orientation(wk2gt_1,133,1025). coords(wk2gt_1,137,39,1025).  visible(wk2gt_1,1025). happensAt(walking(wk2gt_1),1025). orientation(wk2gt_5,150,1025). coords(wk2gt_5,131,125,1025).  visible(wk2gt_5,1025). happensAt(walking(wk2gt_5),1025).
orientation(wk2gt_1,133,1026). coords(wk2gt_1,137,39,1026).  visible(wk2gt_1,1026). happensAt(walking(wk2gt_1),1026). orientation(wk2gt_5,150,1026). coords(wk2gt_5,132,126,1026).  visible(wk2gt_5,1026). happensAt(walking(wk2gt_5),1026).
orientation(wk2gt_1,133,1027). coords(wk2gt_1,138,39,1027).  visible(wk2gt_1,1027). happensAt(walking(wk2gt_1),1027). orientation(wk2gt_5,150,1027). coords(wk2gt_5,134,126,1027).  visible(wk2gt_5,1027). happensAt(walking(wk2gt_5),1027).
orientation(wk2gt_1,133,1028). coords(wk2gt_1,138,38,1028).  visible(wk2gt_1,1028). happensAt(walking(wk2gt_1),1028). orientation(wk2gt_5,150,1028). coords(wk2gt_5,135,127,1028).  visible(wk2gt_5,1028). happensAt(walking(wk2gt_5),1028).
orientation(wk2gt_1,133,1029). coords(wk2gt_1,138,38,1029).  visible(wk2gt_1,1029). happensAt(walking(wk2gt_1),1029). orientation(wk2gt_5,150,1029). coords(wk2gt_5,136,127,1029).  visible(wk2gt_5,1029). happensAt(walking(wk2gt_5),1029).
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
#show holdsAt/2.