
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

	orientation(wk2gt_1,125,451). coords(wk2gt_1,115,53,451).  visible(wk2gt_1,451). happensAt(inactive(wk2gt_1),451). orientation(wk2gt_4,149,451). coords(wk2gt_4,90,99,451).  visible(wk2gt_4,451). happensAt(walking(wk2gt_4),451).
orientation(wk2gt_1,125,452). coords(wk2gt_1,115,53,452).  visible(wk2gt_1,452). happensAt(inactive(wk2gt_1),452). orientation(wk2gt_4,149,452). coords(wk2gt_4,89,98,452).  visible(wk2gt_4,452). happensAt(walking(wk2gt_4),452).
orientation(wk2gt_1,125,453). coords(wk2gt_1,115,53,453).  visible(wk2gt_1,453). happensAt(inactive(wk2gt_1),453). orientation(wk2gt_4,149,453). coords(wk2gt_4,88,98,453).  visible(wk2gt_4,453). happensAt(walking(wk2gt_4),453).
orientation(wk2gt_1,125,454). coords(wk2gt_1,115,53,454).  visible(wk2gt_1,454). happensAt(inactive(wk2gt_1),454). orientation(wk2gt_4,149,454). coords(wk2gt_4,88,98,454).  visible(wk2gt_4,454). happensAt(walking(wk2gt_4),454).
orientation(wk2gt_1,125,455). coords(wk2gt_1,115,53,455).  visible(wk2gt_1,455). happensAt(inactive(wk2gt_1),455). orientation(wk2gt_4,149,455). coords(wk2gt_4,88,98,455).  visible(wk2gt_4,455). happensAt(walking(wk2gt_4),455).
orientation(wk2gt_1,125,456). coords(wk2gt_1,115,53,456).  visible(wk2gt_1,456). happensAt(inactive(wk2gt_1),456). orientation(wk2gt_4,149,456). coords(wk2gt_4,87,98,456).  visible(wk2gt_4,456). happensAt(walking(wk2gt_4),456).
orientation(wk2gt_1,125,457). coords(wk2gt_1,115,53,457).  visible(wk2gt_1,457). happensAt(inactive(wk2gt_1),457). orientation(wk2gt_4,149,457). coords(wk2gt_4,87,98,457).  visible(wk2gt_4,457). happensAt(walking(wk2gt_4),457).
orientation(wk2gt_1,125,458). coords(wk2gt_1,115,53,458).  visible(wk2gt_1,458). happensAt(inactive(wk2gt_1),458). orientation(wk2gt_4,149,458). coords(wk2gt_4,87,98,458).  visible(wk2gt_4,458). happensAt(walking(wk2gt_4),458).
orientation(wk2gt_1,125,459). coords(wk2gt_1,115,53,459).  visible(wk2gt_1,459). happensAt(inactive(wk2gt_1),459). orientation(wk2gt_4,149,459). coords(wk2gt_4,86,98,459).  visible(wk2gt_4,459). happensAt(walking(wk2gt_4),459).
orientation(wk2gt_1,125,460). coords(wk2gt_1,115,53,460).  visible(wk2gt_1,460). happensAt(inactive(wk2gt_1),460). orientation(wk2gt_4,149,460). coords(wk2gt_4,86,98,460).  visible(wk2gt_4,460). happensAt(walking(wk2gt_4),460).
orientation(wk2gt_1,125,461). coords(wk2gt_1,115,53,461).  visible(wk2gt_1,461). happensAt(inactive(wk2gt_1),461). orientation(wk2gt_4,149,461). coords(wk2gt_4,86,98,461).  visible(wk2gt_4,461). happensAt(walking(wk2gt_4),461).
orientation(wk2gt_1,125,462). coords(wk2gt_1,115,53,462).  visible(wk2gt_1,462). happensAt(inactive(wk2gt_1),462). orientation(wk2gt_4,149,462). coords(wk2gt_4,85,98,462).  visible(wk2gt_4,462). happensAt(walking(wk2gt_4),462).
orientation(wk2gt_1,125,463). coords(wk2gt_1,115,53,463).  visible(wk2gt_1,463). happensAt(inactive(wk2gt_1),463). orientation(wk2gt_4,149,463). coords(wk2gt_4,84,98,463).  visible(wk2gt_4,463). happensAt(walking(wk2gt_4),463).
orientation(wk2gt_1,125,464). coords(wk2gt_1,115,53,464).  visible(wk2gt_1,464). happensAt(inactive(wk2gt_1),464). orientation(wk2gt_4,149,464). coords(wk2gt_4,83,98,464).  visible(wk2gt_4,464). happensAt(walking(wk2gt_4),464).
orientation(wk2gt_1,125,465). coords(wk2gt_1,115,53,465).  visible(wk2gt_1,465). happensAt(inactive(wk2gt_1),465). orientation(wk2gt_4,149,465). coords(wk2gt_4,82,98,465).  visible(wk2gt_4,465). happensAt(walking(wk2gt_4),465).
orientation(wk2gt_1,125,466). coords(wk2gt_1,115,53,466).  visible(wk2gt_1,466). happensAt(inactive(wk2gt_1),466). orientation(wk2gt_4,149,466). coords(wk2gt_4,81,98,466).  visible(wk2gt_4,466). happensAt(walking(wk2gt_4),466).
orientation(wk2gt_1,125,467). coords(wk2gt_1,115,53,467).  visible(wk2gt_1,467). happensAt(inactive(wk2gt_1),467). orientation(wk2gt_4,149,467). coords(wk2gt_4,80,98,467).  visible(wk2gt_4,467). happensAt(walking(wk2gt_4),467).
orientation(wk2gt_1,125,468). coords(wk2gt_1,115,53,468).  visible(wk2gt_1,468). happensAt(inactive(wk2gt_1),468). orientation(wk2gt_4,149,468). coords(wk2gt_4,80,98,468).  visible(wk2gt_4,468). happensAt(walking(wk2gt_4),468).
orientation(wk2gt_1,125,469). coords(wk2gt_1,115,53,469).  visible(wk2gt_1,469). happensAt(inactive(wk2gt_1),469). orientation(wk2gt_4,149,469). coords(wk2gt_4,80,98,469).  visible(wk2gt_4,469). happensAt(walking(wk2gt_4),469).
orientation(wk2gt_1,125,470). coords(wk2gt_1,115,53,470).  visible(wk2gt_1,470). happensAt(inactive(wk2gt_1),470). orientation(wk2gt_4,149,470). coords(wk2gt_4,80,98,470).  visible(wk2gt_4,470). happensAt(walking(wk2gt_4),470).
orientation(wk2gt_1,125,471). coords(wk2gt_1,115,53,471).  visible(wk2gt_1,471). happensAt(inactive(wk2gt_1),471). orientation(wk2gt_4,149,471). coords(wk2gt_4,80,98,471).  visible(wk2gt_4,471). happensAt(walking(wk2gt_4),471).
orientation(wk2gt_1,125,472). coords(wk2gt_1,115,53,472).  visible(wk2gt_1,472). happensAt(inactive(wk2gt_1),472). orientation(wk2gt_4,149,472). coords(wk2gt_4,80,98,472).  visible(wk2gt_4,472). happensAt(walking(wk2gt_4),472).
orientation(wk2gt_1,125,473). coords(wk2gt_1,115,53,473).  visible(wk2gt_1,473). happensAt(inactive(wk2gt_1),473). orientation(wk2gt_4,149,473). coords(wk2gt_4,79,98,473).  visible(wk2gt_4,473). happensAt(walking(wk2gt_4),473).
orientation(wk2gt_1,125,474). coords(wk2gt_1,115,53,474).  visible(wk2gt_1,474). happensAt(inactive(wk2gt_1),474). orientation(wk2gt_4,149,474). coords(wk2gt_4,78,98,474).  visible(wk2gt_4,474). happensAt(walking(wk2gt_4),474).
orientation(wk2gt_1,125,475). coords(wk2gt_1,115,53,475).  visible(wk2gt_1,475). happensAt(inactive(wk2gt_1),475). orientation(wk2gt_4,149,475). coords(wk2gt_4,77,98,475).  visible(wk2gt_4,475). happensAt(walking(wk2gt_4),475).
#show holdsAt/2.