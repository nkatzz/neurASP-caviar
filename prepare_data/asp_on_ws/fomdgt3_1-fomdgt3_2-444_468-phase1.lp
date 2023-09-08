
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

	orientation(fomdgt3_1,150,444). coords(fomdgt3_1,46,102,444).  visible(fomdgt3_1,444). happensAt(inactive(fomdgt3_1),444). orientation(fomdgt3_2,42,444). coords(fomdgt3_2,262,96,444).  visible(fomdgt3_2,444). happensAt(inactive(fomdgt3_2),444).
orientation(fomdgt3_1,150,445). coords(fomdgt3_1,46,102,445).  visible(fomdgt3_1,445). happensAt(inactive(fomdgt3_1),445). orientation(fomdgt3_2,42,445). coords(fomdgt3_2,262,96,445).  visible(fomdgt3_2,445). happensAt(inactive(fomdgt3_2),445).
orientation(fomdgt3_1,150,446). coords(fomdgt3_1,46,102,446).  visible(fomdgt3_1,446). happensAt(inactive(fomdgt3_1),446). orientation(fomdgt3_2,42,446). coords(fomdgt3_2,262,96,446).  visible(fomdgt3_2,446). happensAt(inactive(fomdgt3_2),446).
orientation(fomdgt3_1,150,447). coords(fomdgt3_1,46,102,447).  visible(fomdgt3_1,447). happensAt(inactive(fomdgt3_1),447). orientation(fomdgt3_2,42,447). coords(fomdgt3_2,262,96,447).  visible(fomdgt3_2,447). happensAt(inactive(fomdgt3_2),447).
orientation(fomdgt3_1,150,448). coords(fomdgt3_1,46,102,448).  visible(fomdgt3_1,448). happensAt(inactive(fomdgt3_1),448). orientation(fomdgt3_2,42,448). coords(fomdgt3_2,262,96,448).  visible(fomdgt3_2,448). happensAt(inactive(fomdgt3_2),448).
orientation(fomdgt3_1,150,449). coords(fomdgt3_1,46,102,449).  visible(fomdgt3_1,449). happensAt(inactive(fomdgt3_1),449). orientation(fomdgt3_2,42,449). coords(fomdgt3_2,262,96,449).  visible(fomdgt3_2,449). happensAt(inactive(fomdgt3_2),449).
orientation(fomdgt3_1,150,450). coords(fomdgt3_1,46,102,450).  visible(fomdgt3_1,450). happensAt(inactive(fomdgt3_1),450). orientation(fomdgt3_2,42,450). coords(fomdgt3_2,262,96,450).  visible(fomdgt3_2,450). happensAt(inactive(fomdgt3_2),450).
orientation(fomdgt3_1,150,451). coords(fomdgt3_1,46,102,451).  visible(fomdgt3_1,451). happensAt(inactive(fomdgt3_1),451). orientation(fomdgt3_2,42,451). coords(fomdgt3_2,262,96,451).  visible(fomdgt3_2,451). happensAt(inactive(fomdgt3_2),451).
orientation(fomdgt3_1,150,452). coords(fomdgt3_1,46,102,452).  visible(fomdgt3_1,452). happensAt(inactive(fomdgt3_1),452). orientation(fomdgt3_2,42,452). coords(fomdgt3_2,262,96,452).  visible(fomdgt3_2,452). happensAt(inactive(fomdgt3_2),452).
orientation(fomdgt3_1,150,453). coords(fomdgt3_1,46,102,453).  visible(fomdgt3_1,453). happensAt(inactive(fomdgt3_1),453). orientation(fomdgt3_2,42,453). coords(fomdgt3_2,262,96,453).  visible(fomdgt3_2,453). happensAt(inactive(fomdgt3_2),453).
orientation(fomdgt3_1,150,454). coords(fomdgt3_1,46,102,454).  visible(fomdgt3_1,454). happensAt(inactive(fomdgt3_1),454). orientation(fomdgt3_2,42,454). coords(fomdgt3_2,262,96,454).  visible(fomdgt3_2,454). happensAt(inactive(fomdgt3_2),454).
orientation(fomdgt3_1,150,455). coords(fomdgt3_1,46,102,455).  visible(fomdgt3_1,455). happensAt(inactive(fomdgt3_1),455). orientation(fomdgt3_2,42,455). coords(fomdgt3_2,262,96,455).  visible(fomdgt3_2,455). happensAt(inactive(fomdgt3_2),455).
orientation(fomdgt3_1,150,456). coords(fomdgt3_1,46,102,456).  visible(fomdgt3_1,456). happensAt(inactive(fomdgt3_1),456). orientation(fomdgt3_2,42,456). coords(fomdgt3_2,262,96,456).  visible(fomdgt3_2,456). happensAt(inactive(fomdgt3_2),456).
orientation(fomdgt3_1,150,457). coords(fomdgt3_1,46,102,457).  visible(fomdgt3_1,457). happensAt(inactive(fomdgt3_1),457). orientation(fomdgt3_2,42,457). coords(fomdgt3_2,262,96,457).  visible(fomdgt3_2,457). happensAt(inactive(fomdgt3_2),457).
orientation(fomdgt3_1,150,458). coords(fomdgt3_1,46,102,458).  visible(fomdgt3_1,458). happensAt(inactive(fomdgt3_1),458). orientation(fomdgt3_2,42,458). coords(fomdgt3_2,262,96,458).  visible(fomdgt3_2,458). happensAt(inactive(fomdgt3_2),458).
orientation(fomdgt3_1,150,459). coords(fomdgt3_1,46,102,459).  visible(fomdgt3_1,459). happensAt(inactive(fomdgt3_1),459). orientation(fomdgt3_2,42,459). coords(fomdgt3_2,262,96,459).  visible(fomdgt3_2,459). happensAt(inactive(fomdgt3_2),459).
orientation(fomdgt3_1,150,460). coords(fomdgt3_1,46,102,460).  visible(fomdgt3_1,460). happensAt(inactive(fomdgt3_1),460). orientation(fomdgt3_2,42,460). coords(fomdgt3_2,262,96,460).  visible(fomdgt3_2,460). happensAt(inactive(fomdgt3_2),460).
orientation(fomdgt3_1,150,461). coords(fomdgt3_1,46,102,461).  visible(fomdgt3_1,461). happensAt(inactive(fomdgt3_1),461). orientation(fomdgt3_2,42,461). coords(fomdgt3_2,262,96,461).  visible(fomdgt3_2,461). happensAt(inactive(fomdgt3_2),461).
orientation(fomdgt3_1,150,462). coords(fomdgt3_1,46,102,462).  visible(fomdgt3_1,462). happensAt(inactive(fomdgt3_1),462). orientation(fomdgt3_2,42,462). coords(fomdgt3_2,262,96,462).  visible(fomdgt3_2,462). happensAt(inactive(fomdgt3_2),462).
orientation(fomdgt3_1,150,463). coords(fomdgt3_1,46,102,463).  visible(fomdgt3_1,463). happensAt(inactive(fomdgt3_1),463). orientation(fomdgt3_2,42,463). coords(fomdgt3_2,262,96,463).  visible(fomdgt3_2,463). happensAt(inactive(fomdgt3_2),463).
orientation(fomdgt3_1,150,464). coords(fomdgt3_1,46,102,464).  visible(fomdgt3_1,464). happensAt(inactive(fomdgt3_1),464). orientation(fomdgt3_2,42,464). coords(fomdgt3_2,262,96,464).  visible(fomdgt3_2,464). happensAt(inactive(fomdgt3_2),464).
orientation(fomdgt3_1,150,465). coords(fomdgt3_1,46,102,465).  visible(fomdgt3_1,465). happensAt(inactive(fomdgt3_1),465). orientation(fomdgt3_2,42,465). coords(fomdgt3_2,262,96,465).  visible(fomdgt3_2,465). happensAt(inactive(fomdgt3_2),465).
orientation(fomdgt3_1,150,466). coords(fomdgt3_1,46,102,466).  visible(fomdgt3_1,466). happensAt(inactive(fomdgt3_1),466). orientation(fomdgt3_2,42,466). coords(fomdgt3_2,262,96,466).  visible(fomdgt3_2,466). happensAt(inactive(fomdgt3_2),466).
orientation(fomdgt3_1,150,467). coords(fomdgt3_1,46,102,467).  visible(fomdgt3_1,467). happensAt(inactive(fomdgt3_1),467). orientation(fomdgt3_2,42,467). coords(fomdgt3_2,262,96,467).  visible(fomdgt3_2,467). happensAt(inactive(fomdgt3_2),467).
orientation(fomdgt3_1,150,468). coords(fomdgt3_1,46,102,468).  visible(fomdgt3_1,468). happensAt(inactive(fomdgt3_1),468). orientation(fomdgt3_2,42,468). coords(fomdgt3_2,262,96,468).  visible(fomdgt3_2,468). happensAt(inactive(fomdgt3_2),468).
#show holdsAt/2.