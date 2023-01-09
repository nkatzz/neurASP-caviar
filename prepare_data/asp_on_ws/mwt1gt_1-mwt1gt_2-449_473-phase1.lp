
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

	orientation(mwt1gt_1,160,449). coords(mwt1gt_1,72,99,449).  visible(mwt1gt_1,449). happensAt(walking(mwt1gt_1),449). orientation(mwt1gt_2,140,449). coords(mwt1gt_2,72,88,449).  visible(mwt1gt_2,449). happensAt(walking(mwt1gt_2),449).
orientation(mwt1gt_1,160,450). coords(mwt1gt_1,72,99,450).  visible(mwt1gt_1,450). happensAt(walking(mwt1gt_1),450). orientation(mwt1gt_2,140,450). coords(mwt1gt_2,72,88,450).  visible(mwt1gt_2,450). happensAt(walking(mwt1gt_2),450).
orientation(mwt1gt_1,160,451). coords(mwt1gt_1,72,99,451).  visible(mwt1gt_1,451). happensAt(walking(mwt1gt_1),451). orientation(mwt1gt_2,140,451). coords(mwt1gt_2,72,88,451).  visible(mwt1gt_2,451). happensAt(walking(mwt1gt_2),451).
orientation(mwt1gt_1,160,452). coords(mwt1gt_1,72,99,452).  visible(mwt1gt_1,452). happensAt(walking(mwt1gt_1),452). orientation(mwt1gt_2,140,452). coords(mwt1gt_2,72,88,452).  visible(mwt1gt_2,452). happensAt(walking(mwt1gt_2),452).
orientation(mwt1gt_1,160,453). coords(mwt1gt_1,72,97,453).  visible(mwt1gt_1,453). happensAt(walking(mwt1gt_1),453). orientation(mwt1gt_2,140,453). coords(mwt1gt_2,72,87,453).  visible(mwt1gt_2,453). happensAt(walking(mwt1gt_2),453).
orientation(mwt1gt_1,160,454). coords(mwt1gt_1,69,97,454).  visible(mwt1gt_1,454). happensAt(walking(mwt1gt_1),454). orientation(mwt1gt_2,140,454). coords(mwt1gt_2,72,87,454).  visible(mwt1gt_2,454). happensAt(walking(mwt1gt_2),454).
orientation(mwt1gt_1,160,455). coords(mwt1gt_1,69,97,455).  visible(mwt1gt_1,455). happensAt(walking(mwt1gt_1),455). orientation(mwt1gt_2,140,455). coords(mwt1gt_2,73,87,455).  visible(mwt1gt_2,455). happensAt(walking(mwt1gt_2),455).
orientation(mwt1gt_1,160,456). coords(mwt1gt_1,69,97,456).  visible(mwt1gt_1,456). happensAt(walking(mwt1gt_1),456). orientation(mwt1gt_2,140,456). coords(mwt1gt_2,73,87,456).  visible(mwt1gt_2,456). happensAt(walking(mwt1gt_2),456).
orientation(mwt1gt_1,160,457). coords(mwt1gt_1,69,97,457).  visible(mwt1gt_1,457). happensAt(walking(mwt1gt_1),457). orientation(mwt1gt_2,140,457). coords(mwt1gt_2,73,87,457).  visible(mwt1gt_2,457). happensAt(walking(mwt1gt_2),457).
orientation(mwt1gt_1,160,458). coords(mwt1gt_1,69,97,458).  visible(mwt1gt_1,458). happensAt(walking(mwt1gt_1),458). orientation(mwt1gt_2,140,458). coords(mwt1gt_2,73,87,458).  visible(mwt1gt_2,458). happensAt(walking(mwt1gt_2),458).
orientation(mwt1gt_1,160,459). coords(mwt1gt_1,69,97,459).  visible(mwt1gt_1,459). happensAt(walking(mwt1gt_1),459). orientation(mwt1gt_2,140,459). coords(mwt1gt_2,73,87,459).  visible(mwt1gt_2,459). happensAt(walking(mwt1gt_2),459).
orientation(mwt1gt_1,160,460). coords(mwt1gt_1,69,97,460).  visible(mwt1gt_1,460). happensAt(walking(mwt1gt_1),460). orientation(mwt1gt_2,140,460). coords(mwt1gt_2,73,87,460).  visible(mwt1gt_2,460). happensAt(walking(mwt1gt_2),460).
orientation(mwt1gt_1,160,461). coords(mwt1gt_1,69,97,461).  visible(mwt1gt_1,461). happensAt(walking(mwt1gt_1),461). orientation(mwt1gt_2,140,461). coords(mwt1gt_2,73,87,461).  visible(mwt1gt_2,461). happensAt(walking(mwt1gt_2),461).
orientation(mwt1gt_1,160,462). coords(mwt1gt_1,67,97,462).  visible(mwt1gt_1,462). happensAt(walking(mwt1gt_1),462). orientation(mwt1gt_2,140,462). coords(mwt1gt_2,73,83,462).  visible(mwt1gt_2,462). happensAt(walking(mwt1gt_2),462).
orientation(mwt1gt_1,160,463). coords(mwt1gt_1,67,96,463).  visible(mwt1gt_1,463). happensAt(walking(mwt1gt_1),463). orientation(mwt1gt_2,140,463). coords(mwt1gt_2,73,83,463).  visible(mwt1gt_2,463). happensAt(walking(mwt1gt_2),463).
orientation(mwt1gt_1,160,464). coords(mwt1gt_1,67,96,464).  visible(mwt1gt_1,464). happensAt(walking(mwt1gt_1),464). orientation(mwt1gt_2,140,464). coords(mwt1gt_2,73,83,464).  visible(mwt1gt_2,464). happensAt(walking(mwt1gt_2),464).
orientation(mwt1gt_1,160,465). coords(mwt1gt_1,67,96,465).  visible(mwt1gt_1,465). happensAt(walking(mwt1gt_1),465). orientation(mwt1gt_2,140,465). coords(mwt1gt_2,73,83,465).  visible(mwt1gt_2,465). happensAt(walking(mwt1gt_2),465).
orientation(mwt1gt_1,154,466). coords(mwt1gt_1,67,95,466).  visible(mwt1gt_1,466). happensAt(walking(mwt1gt_1),466). orientation(mwt1gt_2,140,466). coords(mwt1gt_2,73,83,466).  visible(mwt1gt_2,466). happensAt(walking(mwt1gt_2),466).
orientation(mwt1gt_1,154,467). coords(mwt1gt_1,67,95,467).  visible(mwt1gt_1,467). happensAt(walking(mwt1gt_1),467). orientation(mwt1gt_2,140,467). coords(mwt1gt_2,73,83,467).  visible(mwt1gt_2,467). happensAt(walking(mwt1gt_2),467).
orientation(mwt1gt_1,154,468). coords(mwt1gt_1,67,95,468).  visible(mwt1gt_1,468). happensAt(walking(mwt1gt_1),468). orientation(mwt1gt_2,140,468). coords(mwt1gt_2,73,79,468).  visible(mwt1gt_2,468). happensAt(walking(mwt1gt_2),468).
orientation(mwt1gt_1,154,469). coords(mwt1gt_1,67,95,469).  visible(mwt1gt_1,469). happensAt(walking(mwt1gt_1),469). orientation(mwt1gt_2,140,469). coords(mwt1gt_2,73,79,469).  visible(mwt1gt_2,469). happensAt(walking(mwt1gt_2),469).
orientation(mwt1gt_1,154,470). coords(mwt1gt_1,67,94,470).  visible(mwt1gt_1,470). happensAt(walking(mwt1gt_1),470). orientation(mwt1gt_2,140,470). coords(mwt1gt_2,73,77,470).  visible(mwt1gt_2,470). happensAt(walking(mwt1gt_2),470).
orientation(mwt1gt_1,154,471). coords(mwt1gt_1,67,94,471).  visible(mwt1gt_1,471). happensAt(walking(mwt1gt_1),471). orientation(mwt1gt_2,140,471). coords(mwt1gt_2,73,77,471).  visible(mwt1gt_2,471). happensAt(walking(mwt1gt_2),471).
orientation(mwt1gt_1,154,472). coords(mwt1gt_1,67,94,472).  visible(mwt1gt_1,472). happensAt(walking(mwt1gt_1),472). orientation(mwt1gt_2,140,472). coords(mwt1gt_2,73,77,472).  visible(mwt1gt_2,472). happensAt(walking(mwt1gt_2),472).
orientation(mwt1gt_1,154,473). coords(mwt1gt_1,66,91,473).  visible(mwt1gt_1,473). happensAt(walking(mwt1gt_1),473). orientation(mwt1gt_2,140,473). coords(mwt1gt_2,72,76,473).  visible(mwt1gt_2,473). happensAt(walking(mwt1gt_2),473).
#show holdsAt/2.