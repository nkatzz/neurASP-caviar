
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

	orientation(wk1gt_1,148,440). coords(wk1gt_1,89,92,440).  visible(wk1gt_1,440). happensAt(walking(wk1gt_1),440). orientation(wk1gt_4,147,440). coords(wk1gt_4,56,169,440).  visible(wk1gt_4,440). happensAt(active(wk1gt_4),440).
orientation(wk1gt_1,148,441). coords(wk1gt_1,89,92,441).  visible(wk1gt_1,441). happensAt(walking(wk1gt_1),441). orientation(wk1gt_4,147,441). coords(wk1gt_4,56,169,441).  visible(wk1gt_4,441). happensAt(active(wk1gt_4),441).
orientation(wk1gt_1,148,442). coords(wk1gt_1,89,92,442).  visible(wk1gt_1,442). happensAt(walking(wk1gt_1),442). orientation(wk1gt_4,147,442). coords(wk1gt_4,56,169,442).  visible(wk1gt_4,442). happensAt(active(wk1gt_4),442).
orientation(wk1gt_1,148,443). coords(wk1gt_1,89,92,443).  visible(wk1gt_1,443). happensAt(walking(wk1gt_1),443). orientation(wk1gt_4,147,443). coords(wk1gt_4,56,169,443).  visible(wk1gt_4,443). happensAt(active(wk1gt_4),443).
orientation(wk1gt_1,148,444). coords(wk1gt_1,89,92,444).  visible(wk1gt_1,444). happensAt(walking(wk1gt_1),444). orientation(wk1gt_4,147,444). coords(wk1gt_4,56,169,444).  visible(wk1gt_4,444). happensAt(active(wk1gt_4),444).
orientation(wk1gt_1,148,445). coords(wk1gt_1,88,91,445).  visible(wk1gt_1,445). happensAt(walking(wk1gt_1),445). orientation(wk1gt_4,147,445). coords(wk1gt_4,56,169,445).  visible(wk1gt_4,445). happensAt(active(wk1gt_4),445).
orientation(wk1gt_1,148,446). coords(wk1gt_1,88,91,446).  visible(wk1gt_1,446). happensAt(walking(wk1gt_1),446). orientation(wk1gt_4,147,446). coords(wk1gt_4,56,169,446).  visible(wk1gt_4,446). happensAt(active(wk1gt_4),446).
orientation(wk1gt_1,148,447). coords(wk1gt_1,88,91,447).  visible(wk1gt_1,447). happensAt(walking(wk1gt_1),447). orientation(wk1gt_4,147,447). coords(wk1gt_4,56,169,447).  visible(wk1gt_4,447). happensAt(active(wk1gt_4),447).
orientation(wk1gt_1,148,448). coords(wk1gt_1,88,91,448).  visible(wk1gt_1,448). happensAt(walking(wk1gt_1),448). orientation(wk1gt_4,147,448). coords(wk1gt_4,56,169,448).  visible(wk1gt_4,448). happensAt(active(wk1gt_4),448).
orientation(wk1gt_1,148,449). coords(wk1gt_1,86,90,449).  visible(wk1gt_1,449). happensAt(walking(wk1gt_1),449). orientation(wk1gt_4,147,449). coords(wk1gt_4,56,169,449).  visible(wk1gt_4,449). happensAt(active(wk1gt_4),449).
orientation(wk1gt_1,148,450). coords(wk1gt_1,86,90,450).  visible(wk1gt_1,450). happensAt(walking(wk1gt_1),450). orientation(wk1gt_4,147,450). coords(wk1gt_4,56,169,450).  visible(wk1gt_4,450). happensAt(active(wk1gt_4),450).
orientation(wk1gt_1,148,451). coords(wk1gt_1,86,90,451).  visible(wk1gt_1,451). happensAt(walking(wk1gt_1),451). orientation(wk1gt_4,147,451). coords(wk1gt_4,56,169,451).  visible(wk1gt_4,451). happensAt(active(wk1gt_4),451).
orientation(wk1gt_1,148,452). coords(wk1gt_1,86,90,452).  visible(wk1gt_1,452). happensAt(walking(wk1gt_1),452). orientation(wk1gt_4,147,452). coords(wk1gt_4,56,169,452).  visible(wk1gt_4,452). happensAt(active(wk1gt_4),452).
orientation(wk1gt_1,148,453). coords(wk1gt_1,86,90,453).  visible(wk1gt_1,453). happensAt(walking(wk1gt_1),453). orientation(wk1gt_4,147,453). coords(wk1gt_4,56,169,453).  visible(wk1gt_4,453). happensAt(active(wk1gt_4),453).
orientation(wk1gt_1,148,454). coords(wk1gt_1,85,88,454).  visible(wk1gt_1,454). happensAt(walking(wk1gt_1),454). orientation(wk1gt_4,147,454). coords(wk1gt_4,56,169,454).  visible(wk1gt_4,454). happensAt(active(wk1gt_4),454).
orientation(wk1gt_1,148,455). coords(wk1gt_1,84,88,455).  visible(wk1gt_1,455). happensAt(walking(wk1gt_1),455). orientation(wk1gt_4,147,455). coords(wk1gt_4,56,169,455).  visible(wk1gt_4,455). happensAt(active(wk1gt_4),455).
orientation(wk1gt_1,148,456). coords(wk1gt_1,84,88,456).  visible(wk1gt_1,456). happensAt(walking(wk1gt_1),456). orientation(wk1gt_4,147,456). coords(wk1gt_4,56,169,456).  visible(wk1gt_4,456). happensAt(active(wk1gt_4),456).
orientation(wk1gt_1,148,457). coords(wk1gt_1,83,88,457).  visible(wk1gt_1,457). happensAt(walking(wk1gt_1),457). orientation(wk1gt_4,147,457). coords(wk1gt_4,56,169,457).  visible(wk1gt_4,457). happensAt(active(wk1gt_4),457).
orientation(wk1gt_1,148,458). coords(wk1gt_1,83,87,458).  visible(wk1gt_1,458). happensAt(walking(wk1gt_1),458). orientation(wk1gt_4,147,458). coords(wk1gt_4,56,169,458).  visible(wk1gt_4,458). happensAt(active(wk1gt_4),458).
orientation(wk1gt_1,148,459). coords(wk1gt_1,83,86,459).  visible(wk1gt_1,459). happensAt(walking(wk1gt_1),459). orientation(wk1gt_4,147,459). coords(wk1gt_4,56,169,459).  visible(wk1gt_4,459). happensAt(active(wk1gt_4),459).
orientation(wk1gt_1,148,460). coords(wk1gt_1,83,86,460).  visible(wk1gt_1,460). happensAt(walking(wk1gt_1),460). orientation(wk1gt_4,147,460). coords(wk1gt_4,56,169,460).  visible(wk1gt_4,460). happensAt(active(wk1gt_4),460).
orientation(wk1gt_1,148,461). coords(wk1gt_1,83,84,461).  visible(wk1gt_1,461). happensAt(walking(wk1gt_1),461). orientation(wk1gt_4,147,461). coords(wk1gt_4,56,169,461).  visible(wk1gt_4,461). happensAt(active(wk1gt_4),461).
orientation(wk1gt_1,148,462). coords(wk1gt_1,83,84,462).  visible(wk1gt_1,462). happensAt(walking(wk1gt_1),462). orientation(wk1gt_4,147,462). coords(wk1gt_4,56,169,462).  visible(wk1gt_4,462). happensAt(active(wk1gt_4),462).
orientation(wk1gt_1,148,463). coords(wk1gt_1,83,83,463).  visible(wk1gt_1,463). happensAt(walking(wk1gt_1),463). orientation(wk1gt_4,147,463). coords(wk1gt_4,56,169,463).  visible(wk1gt_4,463). happensAt(active(wk1gt_4),463).
orientation(wk1gt_1,148,464). coords(wk1gt_1,83,83,464).  visible(wk1gt_1,464). happensAt(walking(wk1gt_1),464). orientation(wk1gt_4,147,464). coords(wk1gt_4,56,169,464).  visible(wk1gt_4,464). happensAt(active(wk1gt_4),464).
#show holdsAt/2.