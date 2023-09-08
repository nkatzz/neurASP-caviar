
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

	orientation(ms3ggt_0,171,446). coords(ms3ggt_0,98,148,446).  visible(ms3ggt_0,446). happensAt(walking(ms3ggt_0),446). orientation(ms3ggt_1,177,446). coords(ms3ggt_1,208,166,446).  visible(ms3ggt_1,446). happensAt(walking(ms3ggt_1),446).
orientation(ms3ggt_0,171,447). coords(ms3ggt_0,97,148,447).  visible(ms3ggt_0,447). happensAt(walking(ms3ggt_0),447). orientation(ms3ggt_1,177,447). coords(ms3ggt_1,209,167,447).  visible(ms3ggt_1,447). happensAt(walking(ms3ggt_1),447).
orientation(ms3ggt_0,171,448). coords(ms3ggt_0,96,148,448).  visible(ms3ggt_0,448). happensAt(walking(ms3ggt_0),448). orientation(ms3ggt_1,177,448). coords(ms3ggt_1,212,168,448).  visible(ms3ggt_1,448). happensAt(walking(ms3ggt_1),448).
orientation(ms3ggt_0,171,449). coords(ms3ggt_0,96,148,449).  visible(ms3ggt_0,449). happensAt(walking(ms3ggt_0),449). orientation(ms3ggt_1,177,449). coords(ms3ggt_1,213,169,449).  visible(ms3ggt_1,449). happensAt(walking(ms3ggt_1),449).
orientation(ms3ggt_0,166,450). coords(ms3ggt_0,87,149,450).  visible(ms3ggt_0,450). happensAt(walking(ms3ggt_0),450). orientation(ms3ggt_1,177,450). coords(ms3ggt_1,214,170,450).  visible(ms3ggt_1,450). happensAt(walking(ms3ggt_1),450).
orientation(ms3ggt_0,166,451). coords(ms3ggt_0,87,149,451).  visible(ms3ggt_0,451). happensAt(walking(ms3ggt_0),451). orientation(ms3ggt_1,177,451). coords(ms3ggt_1,214,171,451).  visible(ms3ggt_1,451). happensAt(walking(ms3ggt_1),451).
orientation(ms3ggt_0,166,452). coords(ms3ggt_0,85,150,452).  visible(ms3ggt_0,452). happensAt(walking(ms3ggt_0),452). orientation(ms3ggt_1,177,452). coords(ms3ggt_1,215,173,452).  visible(ms3ggt_1,452). happensAt(walking(ms3ggt_1),452).
orientation(ms3ggt_0,166,453). coords(ms3ggt_0,85,150,453).  visible(ms3ggt_0,453). happensAt(walking(ms3ggt_0),453). orientation(ms3ggt_1,177,453). coords(ms3ggt_1,215,175,453).  visible(ms3ggt_1,453). happensAt(walking(ms3ggt_1),453).
orientation(ms3ggt_0,166,454). coords(ms3ggt_0,83,151,454).  visible(ms3ggt_0,454). happensAt(walking(ms3ggt_0),454). orientation(ms3ggt_1,177,454). coords(ms3ggt_1,216,177,454).  visible(ms3ggt_1,454). happensAt(walking(ms3ggt_1),454).
orientation(ms3ggt_0,166,455). coords(ms3ggt_0,82,151,455).  visible(ms3ggt_0,455). happensAt(walking(ms3ggt_0),455). orientation(ms3ggt_1,177,455). coords(ms3ggt_1,217,179,455).  visible(ms3ggt_1,455). happensAt(walking(ms3ggt_1),455).
orientation(ms3ggt_0,166,456). coords(ms3ggt_0,82,151,456).  visible(ms3ggt_0,456). happensAt(walking(ms3ggt_0),456). orientation(ms3ggt_1,177,456). coords(ms3ggt_1,218,181,456).  visible(ms3ggt_1,456). happensAt(walking(ms3ggt_1),456).
orientation(ms3ggt_0,166,457). coords(ms3ggt_0,78,152,457).  visible(ms3ggt_0,457). happensAt(walking(ms3ggt_0),457). orientation(ms3ggt_1,177,457). coords(ms3ggt_1,218,185,457).  visible(ms3ggt_1,457). happensAt(walking(ms3ggt_1),457).
orientation(ms3ggt_0,166,458). coords(ms3ggt_0,77,152,458).  visible(ms3ggt_0,458). happensAt(walking(ms3ggt_0),458). orientation(ms3ggt_1,177,458). coords(ms3ggt_1,219,189,458).  visible(ms3ggt_1,458). happensAt(walking(ms3ggt_1),458).
orientation(ms3ggt_0,166,459). coords(ms3ggt_0,86,156,459).  visible(ms3ggt_0,459). happensAt(walking(ms3ggt_0),459). orientation(ms3ggt_1,177,459). coords(ms3ggt_1,221,192,459).  visible(ms3ggt_1,459). happensAt(walking(ms3ggt_1),459).
orientation(ms3ggt_0,166,460). coords(ms3ggt_0,86,157,460).  visible(ms3ggt_0,460). happensAt(walking(ms3ggt_0),460). orientation(ms3ggt_1,170,460). coords(ms3ggt_1,221,193,460).  visible(ms3ggt_1,460). happensAt(walking(ms3ggt_1),460).
orientation(ms3ggt_0,166,461). coords(ms3ggt_0,87,158,461).  visible(ms3ggt_0,461). happensAt(walking(ms3ggt_0),461). orientation(ms3ggt_1,170,461). coords(ms3ggt_1,224,197,461).  visible(ms3ggt_1,461). happensAt(walking(ms3ggt_1),461).
orientation(ms3ggt_0,166,462). coords(ms3ggt_0,86,158,462).  visible(ms3ggt_0,462). happensAt(walking(ms3ggt_0),462). orientation(ms3ggt_1,170,462). coords(ms3ggt_1,225,198,462).  visible(ms3ggt_1,462). happensAt(walking(ms3ggt_1),462).
orientation(ms3ggt_0,166,463). coords(ms3ggt_0,85,161,463).  visible(ms3ggt_0,463). happensAt(walking(ms3ggt_0),463). orientation(ms3ggt_1,170,463). coords(ms3ggt_1,226,198,463).  visible(ms3ggt_1,463). happensAt(walking(ms3ggt_1),463).
orientation(ms3ggt_0,155,464). coords(ms3ggt_0,84,162,464).  visible(ms3ggt_0,464). happensAt(walking(ms3ggt_0),464). orientation(ms3ggt_1,170,464). coords(ms3ggt_1,226,198,464).  visible(ms3ggt_1,464). happensAt(walking(ms3ggt_1),464).
orientation(ms3ggt_0,155,465). coords(ms3ggt_0,85,163,465).  visible(ms3ggt_0,465). happensAt(walking(ms3ggt_0),465). orientation(ms3ggt_1,170,465). coords(ms3ggt_1,228,199,465).  visible(ms3ggt_1,465). happensAt(walking(ms3ggt_1),465).
orientation(ms3ggt_0,160,466). coords(ms3ggt_0,85,166,466).  visible(ms3ggt_0,466). happensAt(walking(ms3ggt_0),466). orientation(ms3ggt_1,174,466). coords(ms3ggt_1,229,201,466).  visible(ms3ggt_1,466). happensAt(walking(ms3ggt_1),466).
orientation(ms3ggt_0,151,467). coords(ms3ggt_0,85,167,467).  visible(ms3ggt_0,467). happensAt(walking(ms3ggt_0),467). orientation(ms3ggt_1,174,467). coords(ms3ggt_1,229,202,467).  visible(ms3ggt_1,467). happensAt(walking(ms3ggt_1),467).
orientation(ms3ggt_0,151,468). coords(ms3ggt_0,84,168,468).  visible(ms3ggt_0,468). happensAt(walking(ms3ggt_0),468). orientation(ms3ggt_1,174,468). coords(ms3ggt_1,229,204,468).  visible(ms3ggt_1,468). happensAt(walking(ms3ggt_1),468).
orientation(ms3ggt_0,151,469). coords(ms3ggt_0,84,168,469).  visible(ms3ggt_0,469). happensAt(walking(ms3ggt_0),469). orientation(ms3ggt_1,174,469). coords(ms3ggt_1,230,206,469).  visible(ms3ggt_1,469). happensAt(walking(ms3ggt_1),469).
orientation(ms3ggt_0,151,470). coords(ms3ggt_0,84,169,470).  visible(ms3ggt_0,470). happensAt(walking(ms3ggt_0),470). orientation(ms3ggt_1,174,470). coords(ms3ggt_1,232,208,470).  visible(ms3ggt_1,470). happensAt(walking(ms3ggt_1),470).
#show holdsAt/2.