
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

	orientation(fomdgt3_6,160,446). coords(fomdgt3_6,49,103,446).  visible(fomdgt3_6,446). happensAt(walking(fomdgt3_6),446). orientation(fomdgt3_7,150,446). coords(fomdgt3_7,91,136,446).  visible(fomdgt3_7,446). happensAt(walking(fomdgt3_7),446).
orientation(fomdgt3_6,160,447). coords(fomdgt3_6,49,103,447).  visible(fomdgt3_6,447). happensAt(walking(fomdgt3_6),447). orientation(fomdgt3_7,150,447). coords(fomdgt3_7,90,136,447).  visible(fomdgt3_7,447). happensAt(walking(fomdgt3_7),447).
orientation(fomdgt3_6,160,448). coords(fomdgt3_6,49,103,448).  visible(fomdgt3_6,448). happensAt(walking(fomdgt3_6),448). orientation(fomdgt3_7,150,448). coords(fomdgt3_7,90,136,448).  visible(fomdgt3_7,448). happensAt(walking(fomdgt3_7),448).
orientation(fomdgt3_6,160,449). coords(fomdgt3_6,49,103,449).  visible(fomdgt3_6,449). happensAt(walking(fomdgt3_6),449). orientation(fomdgt3_7,150,449). coords(fomdgt3_7,90,136,449).  visible(fomdgt3_7,449). happensAt(walking(fomdgt3_7),449).
orientation(fomdgt3_6,160,450). coords(fomdgt3_6,49,103,450).  visible(fomdgt3_6,450). happensAt(walking(fomdgt3_6),450). orientation(fomdgt3_7,150,450). coords(fomdgt3_7,91,135,450).  visible(fomdgt3_7,450). happensAt(walking(fomdgt3_7),450).
orientation(fomdgt3_6,160,451). coords(fomdgt3_6,49,103,451).  visible(fomdgt3_6,451). happensAt(walking(fomdgt3_6),451). orientation(fomdgt3_7,150,451). coords(fomdgt3_7,92,136,451).  visible(fomdgt3_7,451). happensAt(walking(fomdgt3_7),451).
orientation(fomdgt3_6,160,452). coords(fomdgt3_6,49,103,452).  visible(fomdgt3_6,452). happensAt(walking(fomdgt3_6),452). orientation(fomdgt3_7,150,452). coords(fomdgt3_7,94,136,452).  visible(fomdgt3_7,452). happensAt(walking(fomdgt3_7),452).
orientation(fomdgt3_6,160,453). coords(fomdgt3_6,49,103,453).  visible(fomdgt3_6,453). happensAt(walking(fomdgt3_6),453). orientation(fomdgt3_7,150,453). coords(fomdgt3_7,95,136,453).  visible(fomdgt3_7,453). happensAt(walking(fomdgt3_7),453).
orientation(fomdgt3_6,160,454). coords(fomdgt3_6,49,103,454).  visible(fomdgt3_6,454). happensAt(walking(fomdgt3_6),454). orientation(fomdgt3_7,150,454). coords(fomdgt3_7,97,136,454).  visible(fomdgt3_7,454). happensAt(walking(fomdgt3_7),454).
orientation(fomdgt3_6,160,455). coords(fomdgt3_6,49,103,455).  visible(fomdgt3_6,455). happensAt(walking(fomdgt3_6),455). orientation(fomdgt3_7,150,455). coords(fomdgt3_7,97,137,455).  visible(fomdgt3_7,455). happensAt(walking(fomdgt3_7),455).
orientation(fomdgt3_6,160,456). coords(fomdgt3_6,49,103,456).  visible(fomdgt3_6,456). happensAt(walking(fomdgt3_6),456). orientation(fomdgt3_7,150,456). coords(fomdgt3_7,98,137,456).  visible(fomdgt3_7,456). happensAt(walking(fomdgt3_7),456).
orientation(fomdgt3_6,160,457). coords(fomdgt3_6,49,103,457).  visible(fomdgt3_6,457). happensAt(walking(fomdgt3_6),457). orientation(fomdgt3_7,150,457). coords(fomdgt3_7,99,137,457).  visible(fomdgt3_7,457). happensAt(walking(fomdgt3_7),457).
orientation(fomdgt3_6,160,458). coords(fomdgt3_6,49,103,458).  visible(fomdgt3_6,458). happensAt(walking(fomdgt3_6),458). orientation(fomdgt3_7,150,458). coords(fomdgt3_7,100,137,458).  visible(fomdgt3_7,458). happensAt(walking(fomdgt3_7),458).
orientation(fomdgt3_6,160,459). coords(fomdgt3_6,49,103,459).  visible(fomdgt3_6,459). happensAt(walking(fomdgt3_6),459). orientation(fomdgt3_7,150,459). coords(fomdgt3_7,100,137,459).  visible(fomdgt3_7,459). happensAt(walking(fomdgt3_7),459).
orientation(fomdgt3_6,160,460). coords(fomdgt3_6,49,103,460).  visible(fomdgt3_6,460). happensAt(walking(fomdgt3_6),460). orientation(fomdgt3_7,150,460). coords(fomdgt3_7,101,139,460).  visible(fomdgt3_7,460). happensAt(walking(fomdgt3_7),460).
orientation(fomdgt3_6,160,461). coords(fomdgt3_6,49,103,461).  visible(fomdgt3_6,461). happensAt(walking(fomdgt3_6),461). orientation(fomdgt3_7,150,461). coords(fomdgt3_7,101,139,461).  visible(fomdgt3_7,461). happensAt(walking(fomdgt3_7),461).
orientation(fomdgt3_6,160,462). coords(fomdgt3_6,49,103,462).  visible(fomdgt3_6,462). happensAt(walking(fomdgt3_6),462). orientation(fomdgt3_7,150,462). coords(fomdgt3_7,101,139,462).  visible(fomdgt3_7,462). happensAt(walking(fomdgt3_7),462).
orientation(fomdgt3_6,160,463). coords(fomdgt3_6,49,103,463).  visible(fomdgt3_6,463). happensAt(walking(fomdgt3_6),463). orientation(fomdgt3_7,150,463). coords(fomdgt3_7,101,139,463).  visible(fomdgt3_7,463). happensAt(walking(fomdgt3_7),463).
orientation(fomdgt3_6,160,464). coords(fomdgt3_6,49,103,464).  visible(fomdgt3_6,464). happensAt(walking(fomdgt3_6),464). orientation(fomdgt3_7,150,464). coords(fomdgt3_7,101,140,464).  visible(fomdgt3_7,464). happensAt(walking(fomdgt3_7),464).
orientation(fomdgt3_6,160,465). coords(fomdgt3_6,49,103,465).  visible(fomdgt3_6,465). happensAt(walking(fomdgt3_6),465). orientation(fomdgt3_7,150,465). coords(fomdgt3_7,101,140,465).  visible(fomdgt3_7,465). happensAt(walking(fomdgt3_7),465).
orientation(fomdgt3_6,160,466). coords(fomdgt3_6,49,103,466).  visible(fomdgt3_6,466). happensAt(walking(fomdgt3_6),466). orientation(fomdgt3_7,150,466). coords(fomdgt3_7,101,141,466).  visible(fomdgt3_7,466). happensAt(walking(fomdgt3_7),466).
orientation(fomdgt3_6,160,467). coords(fomdgt3_6,49,103,467).  visible(fomdgt3_6,467). happensAt(walking(fomdgt3_6),467). orientation(fomdgt3_7,150,467). coords(fomdgt3_7,103,143,467).  visible(fomdgt3_7,467). happensAt(walking(fomdgt3_7),467).
orientation(fomdgt3_6,160,468). coords(fomdgt3_6,49,103,468).  visible(fomdgt3_6,468). happensAt(walking(fomdgt3_6),468). orientation(fomdgt3_7,150,468). coords(fomdgt3_7,104,144,468).  visible(fomdgt3_7,468). happensAt(walking(fomdgt3_7),468).
orientation(fomdgt3_6,160,469). coords(fomdgt3_6,49,103,469).  visible(fomdgt3_6,469). happensAt(walking(fomdgt3_6),469). orientation(fomdgt3_7,150,469). coords(fomdgt3_7,105,147,469).  visible(fomdgt3_7,469). happensAt(walking(fomdgt3_7),469).
orientation(fomdgt3_6,160,470). coords(fomdgt3_6,49,103,470).  visible(fomdgt3_6,470). happensAt(walking(fomdgt3_6),470). orientation(fomdgt3_7,150,470). coords(fomdgt3_7,106,147,470).  visible(fomdgt3_7,470). happensAt(walking(fomdgt3_7),470).
#show holdsAt/2.