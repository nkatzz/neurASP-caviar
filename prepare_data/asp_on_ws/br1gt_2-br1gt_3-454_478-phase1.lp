
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

	orientation(br1gt_2,12,454). coords(br1gt_2,86,208,454).  visible(br1gt_2,454). happensAt(inactive(br1gt_2),454). orientation(br1gt_3,160,454). coords(br1gt_3,70,146,454).  visible(br1gt_3,454). happensAt(walking(br1gt_3),454).
orientation(br1gt_2,12,455). coords(br1gt_2,86,208,455).  visible(br1gt_2,455). happensAt(inactive(br1gt_2),455). orientation(br1gt_3,160,455). coords(br1gt_3,70,146,455).  visible(br1gt_3,455). happensAt(walking(br1gt_3),455).
orientation(br1gt_2,12,456). coords(br1gt_2,86,208,456).  visible(br1gt_2,456). happensAt(inactive(br1gt_2),456). orientation(br1gt_3,160,456). coords(br1gt_3,71,146,456).  visible(br1gt_3,456). happensAt(walking(br1gt_3),456).
orientation(br1gt_2,12,457). coords(br1gt_2,86,208,457).  visible(br1gt_2,457). happensAt(inactive(br1gt_2),457). orientation(br1gt_3,160,457). coords(br1gt_3,71,146,457).  visible(br1gt_3,457). happensAt(walking(br1gt_3),457).
orientation(br1gt_2,12,458). coords(br1gt_2,86,208,458).  visible(br1gt_2,458). happensAt(inactive(br1gt_2),458). orientation(br1gt_3,160,458). coords(br1gt_3,71,146,458).  visible(br1gt_3,458). happensAt(walking(br1gt_3),458).
orientation(br1gt_2,12,459). coords(br1gt_2,86,208,459).  visible(br1gt_2,459). happensAt(inactive(br1gt_2),459). orientation(br1gt_3,160,459). coords(br1gt_3,72,146,459).  visible(br1gt_3,459). happensAt(walking(br1gt_3),459).
orientation(br1gt_2,12,460). coords(br1gt_2,86,208,460).  visible(br1gt_2,460). happensAt(inactive(br1gt_2),460). orientation(br1gt_3,160,460). coords(br1gt_3,74,146,460).  visible(br1gt_3,460). happensAt(walking(br1gt_3),460).
orientation(br1gt_2,12,461). coords(br1gt_2,86,208,461).  visible(br1gt_2,461). happensAt(inactive(br1gt_2),461). orientation(br1gt_3,160,461). coords(br1gt_3,76,146,461).  visible(br1gt_3,461). happensAt(walking(br1gt_3),461).
orientation(br1gt_2,12,462). coords(br1gt_2,86,208,462).  visible(br1gt_2,462). happensAt(inactive(br1gt_2),462). orientation(br1gt_3,160,462). coords(br1gt_3,77,146,462).  visible(br1gt_3,462). happensAt(walking(br1gt_3),462).
orientation(br1gt_2,12,463). coords(br1gt_2,86,208,463).  visible(br1gt_2,463). happensAt(inactive(br1gt_2),463). orientation(br1gt_3,160,463). coords(br1gt_3,78,146,463).  visible(br1gt_3,463). happensAt(walking(br1gt_3),463).
orientation(br1gt_2,12,464). coords(br1gt_2,86,208,464).  visible(br1gt_2,464). happensAt(inactive(br1gt_2),464). orientation(br1gt_3,160,464). coords(br1gt_3,78,145,464).  visible(br1gt_3,464). happensAt(walking(br1gt_3),464).
orientation(br1gt_2,12,465). coords(br1gt_2,86,208,465).  visible(br1gt_2,465). happensAt(inactive(br1gt_2),465). orientation(br1gt_3,160,465). coords(br1gt_3,79,145,465).  visible(br1gt_3,465). happensAt(walking(br1gt_3),465).
orientation(br1gt_2,12,466). coords(br1gt_2,86,208,466).  visible(br1gt_2,466). happensAt(inactive(br1gt_2),466). orientation(br1gt_3,160,466). coords(br1gt_3,81,145,466).  visible(br1gt_3,466). happensAt(walking(br1gt_3),466).
orientation(br1gt_2,12,467). coords(br1gt_2,86,208,467).  visible(br1gt_2,467). happensAt(inactive(br1gt_2),467). orientation(br1gt_3,160,467). coords(br1gt_3,81,145,467).  visible(br1gt_3,467). happensAt(walking(br1gt_3),467).
orientation(br1gt_2,12,468). coords(br1gt_2,86,208,468).  visible(br1gt_2,468). happensAt(inactive(br1gt_2),468). orientation(br1gt_3,160,468). coords(br1gt_3,81,145,468).  visible(br1gt_3,468). happensAt(walking(br1gt_3),468).
orientation(br1gt_2,12,469). coords(br1gt_2,86,208,469).  visible(br1gt_2,469). happensAt(inactive(br1gt_2),469). orientation(br1gt_3,160,469). coords(br1gt_3,81,144,469).  visible(br1gt_3,469). happensAt(walking(br1gt_3),469).
orientation(br1gt_2,12,470). coords(br1gt_2,86,208,470).  visible(br1gt_2,470). happensAt(inactive(br1gt_2),470). orientation(br1gt_3,160,470). coords(br1gt_3,81,144,470).  visible(br1gt_3,470). happensAt(walking(br1gt_3),470).
orientation(br1gt_2,12,471). coords(br1gt_2,86,208,471).  visible(br1gt_2,471). happensAt(inactive(br1gt_2),471). orientation(br1gt_3,160,471). coords(br1gt_3,81,143,471).  visible(br1gt_3,471). happensAt(walking(br1gt_3),471).
orientation(br1gt_2,12,472). coords(br1gt_2,86,208,472).  visible(br1gt_2,472). happensAt(inactive(br1gt_2),472). orientation(br1gt_3,160,472). coords(br1gt_3,81,143,472).  visible(br1gt_3,472). happensAt(walking(br1gt_3),472).
orientation(br1gt_2,12,473). coords(br1gt_2,86,208,473).  visible(br1gt_2,473). happensAt(inactive(br1gt_2),473). orientation(br1gt_3,160,473). coords(br1gt_3,83,142,473).  visible(br1gt_3,473). happensAt(walking(br1gt_3),473).
orientation(br1gt_2,12,474). coords(br1gt_2,86,208,474).  visible(br1gt_2,474). happensAt(inactive(br1gt_2),474). orientation(br1gt_3,160,474). coords(br1gt_3,85,142,474).  visible(br1gt_3,474). happensAt(walking(br1gt_3),474).
orientation(br1gt_2,12,475). coords(br1gt_2,86,208,475).  visible(br1gt_2,475). happensAt(inactive(br1gt_2),475). orientation(br1gt_3,160,475). coords(br1gt_3,85,141,475).  visible(br1gt_3,475). happensAt(walking(br1gt_3),475).
orientation(br1gt_2,12,476). coords(br1gt_2,86,208,476).  visible(br1gt_2,476). happensAt(inactive(br1gt_2),476). orientation(br1gt_3,160,476). coords(br1gt_3,86,141,476).  visible(br1gt_3,476). happensAt(walking(br1gt_3),476).
orientation(br1gt_2,12,477). coords(br1gt_2,86,208,477).  visible(br1gt_2,477). happensAt(inactive(br1gt_2),477). orientation(br1gt_3,160,477). coords(br1gt_3,87,140,477).  visible(br1gt_3,477). happensAt(walking(br1gt_3),477).
orientation(br1gt_2,12,478). coords(br1gt_2,86,208,478).  visible(br1gt_2,478). happensAt(inactive(br1gt_2),478). orientation(br1gt_3,160,478). coords(br1gt_3,89,140,478).  visible(br1gt_3,478). happensAt(walking(br1gt_3),478).
#show holdsAt/2.