
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

	orientation(wk2gt_2,141,439). coords(wk2gt_2,98,63,439).  visible(wk2gt_2,439). happensAt(inactive(wk2gt_2),439). orientation(wk2gt_4,149,439). coords(wk2gt_4,99,100,439).  visible(wk2gt_4,439). happensAt(walking(wk2gt_4),439).
orientation(wk2gt_2,141,440). coords(wk2gt_2,98,63,440).  visible(wk2gt_2,440). happensAt(inactive(wk2gt_2),440). orientation(wk2gt_4,149,440). coords(wk2gt_4,97,100,440).  visible(wk2gt_4,440). happensAt(walking(wk2gt_4),440).
orientation(wk2gt_2,141,441). coords(wk2gt_2,98,63,441).  visible(wk2gt_2,441). happensAt(inactive(wk2gt_2),441). orientation(wk2gt_4,149,441). coords(wk2gt_4,97,100,441).  visible(wk2gt_4,441). happensAt(walking(wk2gt_4),441).
orientation(wk2gt_2,141,442). coords(wk2gt_2,98,63,442).  visible(wk2gt_2,442). happensAt(inactive(wk2gt_2),442). orientation(wk2gt_4,149,442). coords(wk2gt_4,95,100,442).  visible(wk2gt_4,442). happensAt(walking(wk2gt_4),442).
orientation(wk2gt_2,141,443). coords(wk2gt_2,98,63,443).  visible(wk2gt_2,443). happensAt(inactive(wk2gt_2),443). orientation(wk2gt_4,149,443). coords(wk2gt_4,95,100,443).  visible(wk2gt_4,443). happensAt(walking(wk2gt_4),443).
orientation(wk2gt_2,141,444). coords(wk2gt_2,98,63,444).  visible(wk2gt_2,444). happensAt(inactive(wk2gt_2),444). orientation(wk2gt_4,149,444). coords(wk2gt_4,94,100,444).  visible(wk2gt_4,444). happensAt(walking(wk2gt_4),444).
orientation(wk2gt_2,141,445). coords(wk2gt_2,98,63,445).  visible(wk2gt_2,445). happensAt(inactive(wk2gt_2),445). orientation(wk2gt_4,149,445). coords(wk2gt_4,94,100,445).  visible(wk2gt_4,445). happensAt(walking(wk2gt_4),445).
orientation(wk2gt_2,141,446). coords(wk2gt_2,98,63,446).  visible(wk2gt_2,446). happensAt(inactive(wk2gt_2),446). orientation(wk2gt_4,149,446). coords(wk2gt_4,94,100,446).  visible(wk2gt_4,446). happensAt(walking(wk2gt_4),446).
orientation(wk2gt_2,141,447). coords(wk2gt_2,98,63,447).  visible(wk2gt_2,447). happensAt(inactive(wk2gt_2),447). orientation(wk2gt_4,149,447). coords(wk2gt_4,94,100,447).  visible(wk2gt_4,447). happensAt(walking(wk2gt_4),447).
orientation(wk2gt_2,141,448). coords(wk2gt_2,98,63,448).  visible(wk2gt_2,448). happensAt(inactive(wk2gt_2),448). orientation(wk2gt_4,149,448). coords(wk2gt_4,94,100,448).  visible(wk2gt_4,448). happensAt(walking(wk2gt_4),448).
orientation(wk2gt_2,141,449). coords(wk2gt_2,98,63,449).  visible(wk2gt_2,449). happensAt(inactive(wk2gt_2),449). orientation(wk2gt_4,149,449). coords(wk2gt_4,93,100,449).  visible(wk2gt_4,449). happensAt(walking(wk2gt_4),449).
orientation(wk2gt_2,141,450). coords(wk2gt_2,98,63,450).  visible(wk2gt_2,450). happensAt(inactive(wk2gt_2),450). orientation(wk2gt_4,149,450). coords(wk2gt_4,91,100,450).  visible(wk2gt_4,450). happensAt(walking(wk2gt_4),450).
orientation(wk2gt_2,141,451). coords(wk2gt_2,98,63,451).  visible(wk2gt_2,451). happensAt(inactive(wk2gt_2),451). orientation(wk2gt_4,149,451). coords(wk2gt_4,90,99,451).  visible(wk2gt_4,451). happensAt(walking(wk2gt_4),451).
orientation(wk2gt_2,141,452). coords(wk2gt_2,98,63,452).  visible(wk2gt_2,452). happensAt(inactive(wk2gt_2),452). orientation(wk2gt_4,149,452). coords(wk2gt_4,89,98,452).  visible(wk2gt_4,452). happensAt(walking(wk2gt_4),452).
orientation(wk2gt_2,141,453). coords(wk2gt_2,98,63,453).  visible(wk2gt_2,453). happensAt(inactive(wk2gt_2),453). orientation(wk2gt_4,149,453). coords(wk2gt_4,88,98,453).  visible(wk2gt_4,453). happensAt(walking(wk2gt_4),453).
orientation(wk2gt_2,141,454). coords(wk2gt_2,98,63,454).  visible(wk2gt_2,454). happensAt(inactive(wk2gt_2),454). orientation(wk2gt_4,149,454). coords(wk2gt_4,88,98,454).  visible(wk2gt_4,454). happensAt(walking(wk2gt_4),454).
orientation(wk2gt_2,141,455). coords(wk2gt_2,98,63,455).  visible(wk2gt_2,455). happensAt(inactive(wk2gt_2),455). orientation(wk2gt_4,149,455). coords(wk2gt_4,88,98,455).  visible(wk2gt_4,455). happensAt(walking(wk2gt_4),455).
orientation(wk2gt_2,141,456). coords(wk2gt_2,98,63,456).  visible(wk2gt_2,456). happensAt(inactive(wk2gt_2),456). orientation(wk2gt_4,149,456). coords(wk2gt_4,87,98,456).  visible(wk2gt_4,456). happensAt(walking(wk2gt_4),456).
orientation(wk2gt_2,141,457). coords(wk2gt_2,98,63,457).  visible(wk2gt_2,457). happensAt(inactive(wk2gt_2),457). orientation(wk2gt_4,149,457). coords(wk2gt_4,87,98,457).  visible(wk2gt_4,457). happensAt(walking(wk2gt_4),457).
orientation(wk2gt_2,141,458). coords(wk2gt_2,98,63,458).  visible(wk2gt_2,458). happensAt(inactive(wk2gt_2),458). orientation(wk2gt_4,149,458). coords(wk2gt_4,87,98,458).  visible(wk2gt_4,458). happensAt(walking(wk2gt_4),458).
orientation(wk2gt_2,141,459). coords(wk2gt_2,98,63,459).  visible(wk2gt_2,459). happensAt(inactive(wk2gt_2),459). orientation(wk2gt_4,149,459). coords(wk2gt_4,86,98,459).  visible(wk2gt_4,459). happensAt(walking(wk2gt_4),459).
orientation(wk2gt_2,141,460). coords(wk2gt_2,98,63,460).  visible(wk2gt_2,460). happensAt(inactive(wk2gt_2),460). orientation(wk2gt_4,149,460). coords(wk2gt_4,86,98,460).  visible(wk2gt_4,460). happensAt(walking(wk2gt_4),460).
orientation(wk2gt_2,141,461). coords(wk2gt_2,98,63,461).  visible(wk2gt_2,461). happensAt(inactive(wk2gt_2),461). orientation(wk2gt_4,149,461). coords(wk2gt_4,86,98,461).  visible(wk2gt_4,461). happensAt(walking(wk2gt_4),461).
orientation(wk2gt_2,141,462). coords(wk2gt_2,98,63,462).  visible(wk2gt_2,462). happensAt(inactive(wk2gt_2),462). orientation(wk2gt_4,149,462). coords(wk2gt_4,85,98,462).  visible(wk2gt_4,462). happensAt(walking(wk2gt_4),462).
orientation(wk2gt_2,141,463). coords(wk2gt_2,98,63,463).  visible(wk2gt_2,463). happensAt(inactive(wk2gt_2),463). orientation(wk2gt_4,149,463). coords(wk2gt_4,84,98,463).  visible(wk2gt_4,463). happensAt(walking(wk2gt_4),463).
#show holdsAt/2.