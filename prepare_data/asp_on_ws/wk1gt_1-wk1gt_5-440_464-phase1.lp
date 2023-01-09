
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

	orientation(wk1gt_1,148,440). coords(wk1gt_1,89,92,440).  visible(wk1gt_1,440). happensAt(walking(wk1gt_1),440). orientation(wk1gt_5,162,440). coords(wk1gt_5,52,160,440).  visible(wk1gt_5,440). happensAt(active(wk1gt_5),440).
orientation(wk1gt_1,148,441). coords(wk1gt_1,89,92,441).  visible(wk1gt_1,441). happensAt(walking(wk1gt_1),441). orientation(wk1gt_5,162,441). coords(wk1gt_5,53,160,441).  visible(wk1gt_5,441). happensAt(active(wk1gt_5),441).
orientation(wk1gt_1,148,442). coords(wk1gt_1,89,92,442).  visible(wk1gt_1,442). happensAt(walking(wk1gt_1),442). orientation(wk1gt_5,162,442). coords(wk1gt_5,54,160,442).  visible(wk1gt_5,442). happensAt(active(wk1gt_5),442).
orientation(wk1gt_1,148,443). coords(wk1gt_1,89,92,443).  visible(wk1gt_1,443). happensAt(walking(wk1gt_1),443). orientation(wk1gt_5,162,443). coords(wk1gt_5,54,160,443).  visible(wk1gt_5,443). happensAt(active(wk1gt_5),443).
orientation(wk1gt_1,148,444). coords(wk1gt_1,89,92,444).  visible(wk1gt_1,444). happensAt(walking(wk1gt_1),444). orientation(wk1gt_5,162,444). coords(wk1gt_5,54,160,444).  visible(wk1gt_5,444). happensAt(walking(wk1gt_5),444).
orientation(wk1gt_1,148,445). coords(wk1gt_1,88,91,445).  visible(wk1gt_1,445). happensAt(walking(wk1gt_1),445). orientation(wk1gt_5,162,445). coords(wk1gt_5,54,160,445).  visible(wk1gt_5,445). happensAt(walking(wk1gt_5),445).
orientation(wk1gt_1,148,446). coords(wk1gt_1,88,91,446).  visible(wk1gt_1,446). happensAt(walking(wk1gt_1),446). orientation(wk1gt_5,162,446). coords(wk1gt_5,54,159,446).  visible(wk1gt_5,446). happensAt(walking(wk1gt_5),446).
orientation(wk1gt_1,148,447). coords(wk1gt_1,88,91,447).  visible(wk1gt_1,447). happensAt(walking(wk1gt_1),447). orientation(wk1gt_5,162,447). coords(wk1gt_5,54,159,447).  visible(wk1gt_5,447). happensAt(walking(wk1gt_5),447).
orientation(wk1gt_1,148,448). coords(wk1gt_1,88,91,448).  visible(wk1gt_1,448). happensAt(walking(wk1gt_1),448). orientation(wk1gt_5,162,448). coords(wk1gt_5,54,159,448).  visible(wk1gt_5,448). happensAt(walking(wk1gt_5),448).
orientation(wk1gt_1,148,449). coords(wk1gt_1,86,90,449).  visible(wk1gt_1,449). happensAt(walking(wk1gt_1),449). orientation(wk1gt_5,162,449). coords(wk1gt_5,54,159,449).  visible(wk1gt_5,449). happensAt(walking(wk1gt_5),449).
orientation(wk1gt_1,148,450). coords(wk1gt_1,86,90,450).  visible(wk1gt_1,450). happensAt(walking(wk1gt_1),450). orientation(wk1gt_5,162,450). coords(wk1gt_5,54,158,450).  visible(wk1gt_5,450). happensAt(walking(wk1gt_5),450).
orientation(wk1gt_1,148,451). coords(wk1gt_1,86,90,451).  visible(wk1gt_1,451). happensAt(walking(wk1gt_1),451). orientation(wk1gt_5,162,451). coords(wk1gt_5,54,158,451).  visible(wk1gt_5,451). happensAt(walking(wk1gt_5),451).
orientation(wk1gt_1,148,452). coords(wk1gt_1,86,90,452).  visible(wk1gt_1,452). happensAt(walking(wk1gt_1),452). orientation(wk1gt_5,162,452). coords(wk1gt_5,54,158,452).  visible(wk1gt_5,452). happensAt(walking(wk1gt_5),452).
orientation(wk1gt_1,148,453). coords(wk1gt_1,86,90,453).  visible(wk1gt_1,453). happensAt(walking(wk1gt_1),453). orientation(wk1gt_5,162,453). coords(wk1gt_5,54,158,453).  visible(wk1gt_5,453). happensAt(walking(wk1gt_5),453).
orientation(wk1gt_1,148,454). coords(wk1gt_1,85,88,454).  visible(wk1gt_1,454). happensAt(walking(wk1gt_1),454). orientation(wk1gt_5,162,454). coords(wk1gt_5,54,158,454).  visible(wk1gt_5,454). happensAt(walking(wk1gt_5),454).
orientation(wk1gt_1,148,455). coords(wk1gt_1,84,88,455).  visible(wk1gt_1,455). happensAt(walking(wk1gt_1),455). orientation(wk1gt_5,162,455). coords(wk1gt_5,54,158,455).  visible(wk1gt_5,455). happensAt(walking(wk1gt_5),455).
orientation(wk1gt_1,148,456). coords(wk1gt_1,84,88,456).  visible(wk1gt_1,456). happensAt(walking(wk1gt_1),456). orientation(wk1gt_5,162,456). coords(wk1gt_5,54,158,456).  visible(wk1gt_5,456). happensAt(walking(wk1gt_5),456).
orientation(wk1gt_1,148,457). coords(wk1gt_1,83,88,457).  visible(wk1gt_1,457). happensAt(walking(wk1gt_1),457). orientation(wk1gt_5,162,457). coords(wk1gt_5,54,156,457).  visible(wk1gt_5,457). happensAt(walking(wk1gt_5),457).
orientation(wk1gt_1,148,458). coords(wk1gt_1,83,87,458).  visible(wk1gt_1,458). happensAt(walking(wk1gt_1),458). orientation(wk1gt_5,162,458). coords(wk1gt_5,55,156,458).  visible(wk1gt_5,458). happensAt(walking(wk1gt_5),458).
orientation(wk1gt_1,148,459). coords(wk1gt_1,83,86,459).  visible(wk1gt_1,459). happensAt(walking(wk1gt_1),459). orientation(wk1gt_5,162,459). coords(wk1gt_5,55,156,459).  visible(wk1gt_5,459). happensAt(walking(wk1gt_5),459).
orientation(wk1gt_1,148,460). coords(wk1gt_1,83,86,460).  visible(wk1gt_1,460). happensAt(walking(wk1gt_1),460). orientation(wk1gt_5,162,460). coords(wk1gt_5,55,155,460).  visible(wk1gt_5,460). happensAt(walking(wk1gt_5),460).
orientation(wk1gt_1,148,461). coords(wk1gt_1,83,84,461).  visible(wk1gt_1,461). happensAt(walking(wk1gt_1),461). orientation(wk1gt_5,162,461). coords(wk1gt_5,55,155,461).  visible(wk1gt_5,461). happensAt(walking(wk1gt_5),461).
orientation(wk1gt_1,148,462). coords(wk1gt_1,83,84,462).  visible(wk1gt_1,462). happensAt(walking(wk1gt_1),462). orientation(wk1gt_5,162,462). coords(wk1gt_5,56,154,462).  visible(wk1gt_5,462). happensAt(walking(wk1gt_5),462).
orientation(wk1gt_1,148,463). coords(wk1gt_1,83,83,463).  visible(wk1gt_1,463). happensAt(walking(wk1gt_1),463). orientation(wk1gt_5,162,463). coords(wk1gt_5,56,154,463).  visible(wk1gt_5,463). happensAt(walking(wk1gt_5),463).
orientation(wk1gt_1,148,464). coords(wk1gt_1,83,83,464).  visible(wk1gt_1,464). happensAt(walking(wk1gt_1),464). orientation(wk1gt_5,162,464). coords(wk1gt_5,56,154,464).  visible(wk1gt_5,464). happensAt(walking(wk1gt_5),464).
#show holdsAt/2.