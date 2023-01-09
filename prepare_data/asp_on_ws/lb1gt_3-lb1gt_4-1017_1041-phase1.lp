
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

	orientation(lb1gt_3,134,1017). coords(lb1gt_3,178,78,1017).  visible(lb1gt_3,1017). happensAt(walking(lb1gt_3),1017). orientation(lb1gt_4,90,1017). coords(lb1gt_4,310,112,1017).  visible(lb1gt_4,1017). happensAt(inactive(lb1gt_4),1017).
orientation(lb1gt_3,134,1018). coords(lb1gt_3,177,78,1018).  visible(lb1gt_3,1018). happensAt(walking(lb1gt_3),1018). orientation(lb1gt_4,90,1018). coords(lb1gt_4,310,112,1018).  visible(lb1gt_4,1018). happensAt(inactive(lb1gt_4),1018).
orientation(lb1gt_3,134,1019). coords(lb1gt_3,176,77,1019).  visible(lb1gt_3,1019). happensAt(walking(lb1gt_3),1019). orientation(lb1gt_4,90,1019). coords(lb1gt_4,310,112,1019).  visible(lb1gt_4,1019). happensAt(inactive(lb1gt_4),1019).
orientation(lb1gt_3,134,1020). coords(lb1gt_3,174,77,1020).  visible(lb1gt_3,1020). happensAt(walking(lb1gt_3),1020). orientation(lb1gt_4,90,1020). coords(lb1gt_4,310,112,1020).  visible(lb1gt_4,1020). happensAt(inactive(lb1gt_4),1020).
orientation(lb1gt_3,134,1021). coords(lb1gt_3,172,76,1021).  visible(lb1gt_3,1021). happensAt(walking(lb1gt_3),1021). orientation(lb1gt_4,90,1021). coords(lb1gt_4,310,112,1021).  visible(lb1gt_4,1021). happensAt(inactive(lb1gt_4),1021).
orientation(lb1gt_3,134,1022). coords(lb1gt_3,170,75,1022).  visible(lb1gt_3,1022). happensAt(walking(lb1gt_3),1022). orientation(lb1gt_4,90,1022). coords(lb1gt_4,310,112,1022).  visible(lb1gt_4,1022). happensAt(inactive(lb1gt_4),1022).
orientation(lb1gt_3,134,1023). coords(lb1gt_3,167,74,1023).  visible(lb1gt_3,1023). happensAt(walking(lb1gt_3),1023). orientation(lb1gt_4,90,1023). coords(lb1gt_4,310,112,1023).  visible(lb1gt_4,1023). happensAt(inactive(lb1gt_4),1023).
orientation(lb1gt_3,134,1024). coords(lb1gt_3,166,73,1024).  visible(lb1gt_3,1024). happensAt(walking(lb1gt_3),1024). orientation(lb1gt_4,90,1024). coords(lb1gt_4,310,112,1024).  visible(lb1gt_4,1024). happensAt(inactive(lb1gt_4),1024).
orientation(lb1gt_3,134,1025). coords(lb1gt_3,166,71,1025).  visible(lb1gt_3,1025). happensAt(walking(lb1gt_3),1025). orientation(lb1gt_4,90,1025). coords(lb1gt_4,310,112,1025).  visible(lb1gt_4,1025). happensAt(inactive(lb1gt_4),1025).
orientation(lb1gt_3,134,1026). coords(lb1gt_3,165,70,1026).  visible(lb1gt_3,1026). happensAt(walking(lb1gt_3),1026). orientation(lb1gt_4,90,1026). coords(lb1gt_4,310,112,1026).  visible(lb1gt_4,1026). happensAt(inactive(lb1gt_4),1026).
orientation(lb1gt_3,134,1027). coords(lb1gt_3,164,69,1027).  visible(lb1gt_3,1027). happensAt(walking(lb1gt_3),1027). orientation(lb1gt_4,90,1027). coords(lb1gt_4,310,112,1027).  visible(lb1gt_4,1027). happensAt(inactive(lb1gt_4),1027).
orientation(lb1gt_3,134,1028). coords(lb1gt_3,164,69,1028).  visible(lb1gt_3,1028). happensAt(walking(lb1gt_3),1028). orientation(lb1gt_4,90,1028). coords(lb1gt_4,310,112,1028).  visible(lb1gt_4,1028). happensAt(inactive(lb1gt_4),1028).
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
#show holdsAt/2.