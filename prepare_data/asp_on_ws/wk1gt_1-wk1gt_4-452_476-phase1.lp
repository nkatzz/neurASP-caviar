
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
orientation(wk1gt_1,143,465). coords(wk1gt_1,82,83,465).  visible(wk1gt_1,465). happensAt(walking(wk1gt_1),465). orientation(wk1gt_4,147,465). coords(wk1gt_4,56,169,465).  visible(wk1gt_4,465). happensAt(active(wk1gt_4),465).
orientation(wk1gt_1,143,466). coords(wk1gt_1,82,83,466).  visible(wk1gt_1,466). happensAt(walking(wk1gt_1),466). orientation(wk1gt_4,147,466). coords(wk1gt_4,56,169,466).  visible(wk1gt_4,466). happensAt(active(wk1gt_4),466).
orientation(wk1gt_1,143,467). coords(wk1gt_1,82,82,467).  visible(wk1gt_1,467). happensAt(walking(wk1gt_1),467). orientation(wk1gt_4,147,467). coords(wk1gt_4,56,169,467).  visible(wk1gt_4,467). happensAt(active(wk1gt_4),467).
orientation(wk1gt_1,143,468). coords(wk1gt_1,82,82,468).  visible(wk1gt_1,468). happensAt(walking(wk1gt_1),468). orientation(wk1gt_4,147,468). coords(wk1gt_4,56,169,468).  visible(wk1gt_4,468). happensAt(active(wk1gt_4),468).
orientation(wk1gt_1,143,469). coords(wk1gt_1,82,81,469).  visible(wk1gt_1,469). happensAt(walking(wk1gt_1),469). orientation(wk1gt_4,147,469). coords(wk1gt_4,56,168,469).  visible(wk1gt_4,469). happensAt(active(wk1gt_4),469).
orientation(wk1gt_1,143,470). coords(wk1gt_1,82,81,470).  visible(wk1gt_1,470). happensAt(walking(wk1gt_1),470). orientation(wk1gt_4,147,470). coords(wk1gt_4,56,168,470).  visible(wk1gt_4,470). happensAt(active(wk1gt_4),470).
orientation(wk1gt_1,147,471). coords(wk1gt_1,82,80,471).  visible(wk1gt_1,471). happensAt(walking(wk1gt_1),471). orientation(wk1gt_4,147,471). coords(wk1gt_4,56,168,471).  visible(wk1gt_4,471). happensAt(active(wk1gt_4),471).
orientation(wk1gt_1,147,472). coords(wk1gt_1,81,79,472).  visible(wk1gt_1,472). happensAt(walking(wk1gt_1),472). orientation(wk1gt_4,147,472). coords(wk1gt_4,56,168,472).  visible(wk1gt_4,472). happensAt(active(wk1gt_4),472).
orientation(wk1gt_1,147,473). coords(wk1gt_1,81,79,473).  visible(wk1gt_1,473). happensAt(walking(wk1gt_1),473). orientation(wk1gt_4,147,473). coords(wk1gt_4,56,168,473).  visible(wk1gt_4,473). happensAt(active(wk1gt_4),473).
orientation(wk1gt_1,147,474). coords(wk1gt_1,81,79,474).  visible(wk1gt_1,474). happensAt(walking(wk1gt_1),474). orientation(wk1gt_4,147,474). coords(wk1gt_4,56,168,474).  visible(wk1gt_4,474). happensAt(active(wk1gt_4),474).
orientation(wk1gt_1,147,475). coords(wk1gt_1,81,79,475).  visible(wk1gt_1,475). happensAt(walking(wk1gt_1),475). orientation(wk1gt_4,147,475). coords(wk1gt_4,56,168,475).  visible(wk1gt_4,475). happensAt(active(wk1gt_4),475).
orientation(wk1gt_1,147,476). coords(wk1gt_1,81,79,476).  visible(wk1gt_1,476). happensAt(walking(wk1gt_1),476). orientation(wk1gt_4,147,476). coords(wk1gt_4,56,168,476).  visible(wk1gt_4,476). happensAt(active(wk1gt_4),476).
#show holdsAt/2.