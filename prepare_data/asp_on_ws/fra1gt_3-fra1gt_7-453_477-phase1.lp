
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

	orientation(fra1gt_3,90,453). coords(fra1gt_3,104,213,453).  happensAt(appear(fra1gt_3),453). happensAt(active(fra1gt_3),453). orientation(fra1gt_7,146,453). coords(fra1gt_7,55,88,453).  visible(fra1gt_7,453). happensAt(walking(fra1gt_7),453).
orientation(fra1gt_3,90,454). coords(fra1gt_3,104,213,454).  visible(fra1gt_3,454). happensAt(active(fra1gt_3),454). orientation(fra1gt_7,146,454). coords(fra1gt_7,55,88,454).  visible(fra1gt_7,454). happensAt(walking(fra1gt_7),454).
orientation(fra1gt_3,90,455). coords(fra1gt_3,104,213,455).  visible(fra1gt_3,455). happensAt(active(fra1gt_3),455). orientation(fra1gt_7,146,455). coords(fra1gt_7,55,89,455).  visible(fra1gt_7,455). happensAt(walking(fra1gt_7),455).
orientation(fra1gt_3,90,456). coords(fra1gt_3,104,213,456).  visible(fra1gt_3,456). happensAt(active(fra1gt_3),456). orientation(fra1gt_7,146,456). coords(fra1gt_7,54,90,456).  visible(fra1gt_7,456). happensAt(walking(fra1gt_7),456).
orientation(fra1gt_3,175,457). coords(fra1gt_3,103,212,457).  visible(fra1gt_3,457). happensAt(active(fra1gt_3),457). orientation(fra1gt_7,146,457). coords(fra1gt_7,54,90,457).  visible(fra1gt_7,457). happensAt(walking(fra1gt_7),457).
orientation(fra1gt_3,175,458). coords(fra1gt_3,103,212,458).  visible(fra1gt_3,458). happensAt(active(fra1gt_3),458). orientation(fra1gt_7,146,458). coords(fra1gt_7,54,90,458).  visible(fra1gt_7,458). happensAt(walking(fra1gt_7),458).
orientation(fra1gt_3,175,459). coords(fra1gt_3,103,212,459).  visible(fra1gt_3,459). happensAt(active(fra1gt_3),459). orientation(fra1gt_7,146,459). coords(fra1gt_7,54,90,459).  visible(fra1gt_7,459). happensAt(walking(fra1gt_7),459).
orientation(fra1gt_3,175,460). coords(fra1gt_3,103,212,460).  visible(fra1gt_3,460). happensAt(active(fra1gt_3),460). orientation(fra1gt_7,146,460). coords(fra1gt_7,54,90,460).  visible(fra1gt_7,460). happensAt(walking(fra1gt_7),460).
orientation(fra1gt_3,175,461). coords(fra1gt_3,103,212,461).  visible(fra1gt_3,461). happensAt(active(fra1gt_3),461). orientation(fra1gt_7,146,461). coords(fra1gt_7,53,90,461).  visible(fra1gt_7,461). happensAt(walking(fra1gt_7),461).
orientation(fra1gt_3,175,462). coords(fra1gt_3,103,211,462).  visible(fra1gt_3,462). happensAt(active(fra1gt_3),462). orientation(fra1gt_7,146,462). coords(fra1gt_7,53,90,462).  visible(fra1gt_7,462). happensAt(walking(fra1gt_7),462).
orientation(fra1gt_3,175,463). coords(fra1gt_3,103,211,463).  visible(fra1gt_3,463). happensAt(active(fra1gt_3),463). orientation(fra1gt_7,146,463). coords(fra1gt_7,53,90,463).  visible(fra1gt_7,463). happensAt(walking(fra1gt_7),463).
orientation(fra1gt_3,175,464). coords(fra1gt_3,103,210,464).  visible(fra1gt_3,464). happensAt(active(fra1gt_3),464). orientation(fra1gt_7,146,464). coords(fra1gt_7,51,90,464).  visible(fra1gt_7,464). happensAt(walking(fra1gt_7),464).
orientation(fra1gt_3,175,465). coords(fra1gt_3,103,209,465).  visible(fra1gt_3,465). happensAt(active(fra1gt_3),465). orientation(fra1gt_7,146,465). coords(fra1gt_7,51,90,465).  visible(fra1gt_7,465). happensAt(walking(fra1gt_7),465).
orientation(fra1gt_3,175,466). coords(fra1gt_3,103,209,466).  visible(fra1gt_3,466). happensAt(active(fra1gt_3),466). orientation(fra1gt_7,146,466). coords(fra1gt_7,51,90,466).  visible(fra1gt_7,466). happensAt(walking(fra1gt_7),466).
orientation(fra1gt_3,175,467). coords(fra1gt_3,103,208,467).  visible(fra1gt_3,467). happensAt(active(fra1gt_3),467). orientation(fra1gt_7,146,467). coords(fra1gt_7,50,90,467).  visible(fra1gt_7,467). happensAt(walking(fra1gt_7),467).
orientation(fra1gt_3,175,468). coords(fra1gt_3,103,207,468).  visible(fra1gt_3,468). happensAt(active(fra1gt_3),468). orientation(fra1gt_7,146,468). coords(fra1gt_7,50,90,468).  visible(fra1gt_7,468). happensAt(walking(fra1gt_7),468).
orientation(fra1gt_3,175,469). coords(fra1gt_3,103,207,469).  visible(fra1gt_3,469). happensAt(active(fra1gt_3),469). orientation(fra1gt_7,146,469). coords(fra1gt_7,50,90,469).  visible(fra1gt_7,469). happensAt(walking(fra1gt_7),469).
orientation(fra1gt_3,175,470). coords(fra1gt_3,103,207,470).  visible(fra1gt_3,470). happensAt(active(fra1gt_3),470). orientation(fra1gt_7,146,470). coords(fra1gt_7,50,90,470).  visible(fra1gt_7,470). happensAt(walking(fra1gt_7),470).
orientation(fra1gt_3,175,471). coords(fra1gt_3,103,207,471).  visible(fra1gt_3,471). happensAt(active(fra1gt_3),471). orientation(fra1gt_7,146,471). coords(fra1gt_7,50,90,471).  visible(fra1gt_7,471). happensAt(walking(fra1gt_7),471).
orientation(fra1gt_3,175,472). coords(fra1gt_3,103,207,472).  visible(fra1gt_3,472). happensAt(active(fra1gt_3),472). orientation(fra1gt_7,146,472). coords(fra1gt_7,50,90,472).  visible(fra1gt_7,472). happensAt(walking(fra1gt_7),472).
orientation(fra1gt_3,175,473). coords(fra1gt_3,103,207,473).  visible(fra1gt_3,473). happensAt(active(fra1gt_3),473). orientation(fra1gt_7,146,473). coords(fra1gt_7,50,90,473).  visible(fra1gt_7,473). happensAt(walking(fra1gt_7),473).
orientation(fra1gt_3,175,474). coords(fra1gt_3,103,207,474).  visible(fra1gt_3,474). happensAt(active(fra1gt_3),474). orientation(fra1gt_7,146,474). coords(fra1gt_7,50,90,474).  visible(fra1gt_7,474). happensAt(walking(fra1gt_7),474).
orientation(fra1gt_3,175,475). coords(fra1gt_3,103,206,475).  visible(fra1gt_3,475). happensAt(active(fra1gt_3),475). orientation(fra1gt_7,146,475). coords(fra1gt_7,50,90,475).  visible(fra1gt_7,475). happensAt(walking(fra1gt_7),475).
orientation(fra1gt_3,175,476). coords(fra1gt_3,102,205,476).  visible(fra1gt_3,476). happensAt(active(fra1gt_3),476). orientation(fra1gt_7,146,476). coords(fra1gt_7,49,90,476).  visible(fra1gt_7,476). happensAt(walking(fra1gt_7),476).
orientation(fra1gt_3,175,477). coords(fra1gt_3,102,205,477).  visible(fra1gt_3,477). happensAt(active(fra1gt_3),477). orientation(fra1gt_7,146,477). coords(fra1gt_7,49,90,477).  visible(fra1gt_7,477). happensAt(walking(fra1gt_7),477).
#show holdsAt/2.