
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
orientation(wk1gt_1,143,465). coords(wk1gt_1,82,83,465).  visible(wk1gt_1,465). happensAt(walking(wk1gt_1),465). orientation(wk1gt_5,162,465). coords(wk1gt_5,56,153,465).  visible(wk1gt_5,465). happensAt(walking(wk1gt_5),465).
orientation(wk1gt_1,143,466). coords(wk1gt_1,82,83,466).  visible(wk1gt_1,466). happensAt(walking(wk1gt_1),466). orientation(wk1gt_5,162,466). coords(wk1gt_5,56,153,466).  visible(wk1gt_5,466). happensAt(walking(wk1gt_5),466).
orientation(wk1gt_1,143,467). coords(wk1gt_1,82,82,467).  visible(wk1gt_1,467). happensAt(walking(wk1gt_1),467). orientation(wk1gt_5,162,467). coords(wk1gt_5,56,153,467).  visible(wk1gt_5,467). happensAt(walking(wk1gt_5),467).
orientation(wk1gt_1,143,468). coords(wk1gt_1,82,82,468).  visible(wk1gt_1,468). happensAt(walking(wk1gt_1),468). orientation(wk1gt_5,162,468). coords(wk1gt_5,56,153,468).  visible(wk1gt_5,468). happensAt(walking(wk1gt_5),468).
orientation(wk1gt_1,143,469). coords(wk1gt_1,82,81,469).  visible(wk1gt_1,469). happensAt(walking(wk1gt_1),469). orientation(wk1gt_5,162,469). coords(wk1gt_5,56,153,469).  visible(wk1gt_5,469). happensAt(walking(wk1gt_5),469).
orientation(wk1gt_1,143,470). coords(wk1gt_1,82,81,470).  visible(wk1gt_1,470). happensAt(walking(wk1gt_1),470). orientation(wk1gt_5,162,470). coords(wk1gt_5,56,153,470).  visible(wk1gt_5,470). happensAt(walking(wk1gt_5),470).
orientation(wk1gt_1,147,471). coords(wk1gt_1,82,80,471).  visible(wk1gt_1,471). happensAt(walking(wk1gt_1),471). orientation(wk1gt_5,162,471). coords(wk1gt_5,57,153,471).  visible(wk1gt_5,471). happensAt(walking(wk1gt_5),471).
orientation(wk1gt_1,147,472). coords(wk1gt_1,81,79,472).  visible(wk1gt_1,472). happensAt(walking(wk1gt_1),472). orientation(wk1gt_5,162,472). coords(wk1gt_5,57,153,472).  visible(wk1gt_5,472). happensAt(walking(wk1gt_5),472).
orientation(wk1gt_1,147,473). coords(wk1gt_1,81,79,473).  visible(wk1gt_1,473). happensAt(walking(wk1gt_1),473). orientation(wk1gt_5,162,473). coords(wk1gt_5,57,153,473).  visible(wk1gt_5,473). happensAt(walking(wk1gt_5),473).
orientation(wk1gt_1,147,474). coords(wk1gt_1,81,79,474).  visible(wk1gt_1,474). happensAt(walking(wk1gt_1),474). orientation(wk1gt_5,162,474). coords(wk1gt_5,57,153,474).  visible(wk1gt_5,474). happensAt(walking(wk1gt_5),474).
orientation(wk1gt_1,147,475). coords(wk1gt_1,81,79,475).  visible(wk1gt_1,475). happensAt(walking(wk1gt_1),475). orientation(wk1gt_5,162,475). coords(wk1gt_5,57,153,475).  visible(wk1gt_5,475). happensAt(walking(wk1gt_5),475).
orientation(wk1gt_1,147,476). coords(wk1gt_1,81,79,476).  visible(wk1gt_1,476). happensAt(walking(wk1gt_1),476). orientation(wk1gt_5,162,476). coords(wk1gt_5,57,152,476).  visible(wk1gt_5,476). happensAt(active(wk1gt_5),476).
#show holdsAt/2.