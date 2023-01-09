
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

	orientation(fomdgt3_2,42,438). coords(fomdgt3_2,262,96,438).  visible(fomdgt3_2,438). happensAt(inactive(fomdgt3_2),438). orientation(fomdgt3_6,160,438). coords(fomdgt3_6,49,97,438).  visible(fomdgt3_6,438). happensAt(walking(fomdgt3_6),438).
orientation(fomdgt3_2,42,439). coords(fomdgt3_2,262,96,439).  visible(fomdgt3_2,439). happensAt(inactive(fomdgt3_2),439). orientation(fomdgt3_6,160,439). coords(fomdgt3_6,49,103,439).  visible(fomdgt3_6,439). happensAt(walking(fomdgt3_6),439).
orientation(fomdgt3_2,42,440). coords(fomdgt3_2,262,96,440).  visible(fomdgt3_2,440). happensAt(inactive(fomdgt3_2),440). orientation(fomdgt3_6,160,440). coords(fomdgt3_6,49,103,440).  visible(fomdgt3_6,440). happensAt(walking(fomdgt3_6),440).
orientation(fomdgt3_2,42,441). coords(fomdgt3_2,262,96,441).  visible(fomdgt3_2,441). happensAt(inactive(fomdgt3_2),441). orientation(fomdgt3_6,160,441). coords(fomdgt3_6,49,103,441).  visible(fomdgt3_6,441). happensAt(walking(fomdgt3_6),441).
orientation(fomdgt3_2,42,442). coords(fomdgt3_2,262,96,442).  visible(fomdgt3_2,442). happensAt(inactive(fomdgt3_2),442). orientation(fomdgt3_6,160,442). coords(fomdgt3_6,49,103,442).  visible(fomdgt3_6,442). happensAt(walking(fomdgt3_6),442).
orientation(fomdgt3_2,42,443). coords(fomdgt3_2,262,96,443).  visible(fomdgt3_2,443). happensAt(inactive(fomdgt3_2),443). orientation(fomdgt3_6,160,443). coords(fomdgt3_6,49,103,443).  visible(fomdgt3_6,443). happensAt(walking(fomdgt3_6),443).
orientation(fomdgt3_2,42,444). coords(fomdgt3_2,262,96,444).  visible(fomdgt3_2,444). happensAt(inactive(fomdgt3_2),444). orientation(fomdgt3_6,160,444). coords(fomdgt3_6,49,103,444).  visible(fomdgt3_6,444). happensAt(walking(fomdgt3_6),444).
orientation(fomdgt3_2,42,445). coords(fomdgt3_2,262,96,445).  visible(fomdgt3_2,445). happensAt(inactive(fomdgt3_2),445). orientation(fomdgt3_6,160,445). coords(fomdgt3_6,49,103,445).  visible(fomdgt3_6,445). happensAt(walking(fomdgt3_6),445).
orientation(fomdgt3_2,42,446). coords(fomdgt3_2,262,96,446).  visible(fomdgt3_2,446). happensAt(inactive(fomdgt3_2),446). orientation(fomdgt3_6,160,446). coords(fomdgt3_6,49,103,446).  visible(fomdgt3_6,446). happensAt(walking(fomdgt3_6),446).
orientation(fomdgt3_2,42,447). coords(fomdgt3_2,262,96,447).  visible(fomdgt3_2,447). happensAt(inactive(fomdgt3_2),447). orientation(fomdgt3_6,160,447). coords(fomdgt3_6,49,103,447).  visible(fomdgt3_6,447). happensAt(walking(fomdgt3_6),447).
orientation(fomdgt3_2,42,448). coords(fomdgt3_2,262,96,448).  visible(fomdgt3_2,448). happensAt(inactive(fomdgt3_2),448). orientation(fomdgt3_6,160,448). coords(fomdgt3_6,49,103,448).  visible(fomdgt3_6,448). happensAt(walking(fomdgt3_6),448).
orientation(fomdgt3_2,42,449). coords(fomdgt3_2,262,96,449).  visible(fomdgt3_2,449). happensAt(inactive(fomdgt3_2),449). orientation(fomdgt3_6,160,449). coords(fomdgt3_6,49,103,449).  visible(fomdgt3_6,449). happensAt(walking(fomdgt3_6),449).
orientation(fomdgt3_2,42,450). coords(fomdgt3_2,262,96,450).  visible(fomdgt3_2,450). happensAt(inactive(fomdgt3_2),450). orientation(fomdgt3_6,160,450). coords(fomdgt3_6,49,103,450).  visible(fomdgt3_6,450). happensAt(walking(fomdgt3_6),450).
orientation(fomdgt3_2,42,451). coords(fomdgt3_2,262,96,451).  visible(fomdgt3_2,451). happensAt(inactive(fomdgt3_2),451). orientation(fomdgt3_6,160,451). coords(fomdgt3_6,49,103,451).  visible(fomdgt3_6,451). happensAt(walking(fomdgt3_6),451).
orientation(fomdgt3_2,42,452). coords(fomdgt3_2,262,96,452).  visible(fomdgt3_2,452). happensAt(inactive(fomdgt3_2),452). orientation(fomdgt3_6,160,452). coords(fomdgt3_6,49,103,452).  visible(fomdgt3_6,452). happensAt(walking(fomdgt3_6),452).
orientation(fomdgt3_2,42,453). coords(fomdgt3_2,262,96,453).  visible(fomdgt3_2,453). happensAt(inactive(fomdgt3_2),453). orientation(fomdgt3_6,160,453). coords(fomdgt3_6,49,103,453).  visible(fomdgt3_6,453). happensAt(walking(fomdgt3_6),453).
orientation(fomdgt3_2,42,454). coords(fomdgt3_2,262,96,454).  visible(fomdgt3_2,454). happensAt(inactive(fomdgt3_2),454). orientation(fomdgt3_6,160,454). coords(fomdgt3_6,49,103,454).  visible(fomdgt3_6,454). happensAt(walking(fomdgt3_6),454).
orientation(fomdgt3_2,42,455). coords(fomdgt3_2,262,96,455).  visible(fomdgt3_2,455). happensAt(inactive(fomdgt3_2),455). orientation(fomdgt3_6,160,455). coords(fomdgt3_6,49,103,455).  visible(fomdgt3_6,455). happensAt(walking(fomdgt3_6),455).
orientation(fomdgt3_2,42,456). coords(fomdgt3_2,262,96,456).  visible(fomdgt3_2,456). happensAt(inactive(fomdgt3_2),456). orientation(fomdgt3_6,160,456). coords(fomdgt3_6,49,103,456).  visible(fomdgt3_6,456). happensAt(walking(fomdgt3_6),456).
orientation(fomdgt3_2,42,457). coords(fomdgt3_2,262,96,457).  visible(fomdgt3_2,457). happensAt(inactive(fomdgt3_2),457). orientation(fomdgt3_6,160,457). coords(fomdgt3_6,49,103,457).  visible(fomdgt3_6,457). happensAt(walking(fomdgt3_6),457).
orientation(fomdgt3_2,42,458). coords(fomdgt3_2,262,96,458).  visible(fomdgt3_2,458). happensAt(inactive(fomdgt3_2),458). orientation(fomdgt3_6,160,458). coords(fomdgt3_6,49,103,458).  visible(fomdgt3_6,458). happensAt(walking(fomdgt3_6),458).
orientation(fomdgt3_2,42,459). coords(fomdgt3_2,262,96,459).  visible(fomdgt3_2,459). happensAt(inactive(fomdgt3_2),459). orientation(fomdgt3_6,160,459). coords(fomdgt3_6,49,103,459).  visible(fomdgt3_6,459). happensAt(walking(fomdgt3_6),459).
orientation(fomdgt3_2,42,460). coords(fomdgt3_2,262,96,460).  visible(fomdgt3_2,460). happensAt(inactive(fomdgt3_2),460). orientation(fomdgt3_6,160,460). coords(fomdgt3_6,49,103,460).  visible(fomdgt3_6,460). happensAt(walking(fomdgt3_6),460).
orientation(fomdgt3_2,42,461). coords(fomdgt3_2,262,96,461).  visible(fomdgt3_2,461). happensAt(inactive(fomdgt3_2),461). orientation(fomdgt3_6,160,461). coords(fomdgt3_6,49,103,461).  visible(fomdgt3_6,461). happensAt(walking(fomdgt3_6),461).
orientation(fomdgt3_2,42,462). coords(fomdgt3_2,262,96,462).  visible(fomdgt3_2,462). happensAt(inactive(fomdgt3_2),462). orientation(fomdgt3_6,160,462). coords(fomdgt3_6,49,103,462).  visible(fomdgt3_6,462). happensAt(walking(fomdgt3_6),462).
#show holdsAt/2.