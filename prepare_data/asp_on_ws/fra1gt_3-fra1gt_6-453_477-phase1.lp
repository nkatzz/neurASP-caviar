
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

	orientation(fra1gt_3,90,453). coords(fra1gt_3,104,213,453).  happensAt(appear(fra1gt_3),453). happensAt(active(fra1gt_3),453). orientation(fra1gt_6,166,453). coords(fra1gt_6,44,180,453).  visible(fra1gt_6,453). happensAt(running(fra1gt_6),453).
orientation(fra1gt_3,90,454). coords(fra1gt_3,104,213,454).  visible(fra1gt_3,454). happensAt(active(fra1gt_3),454). orientation(fra1gt_6,166,454). coords(fra1gt_6,42,182,454).  visible(fra1gt_6,454). happensAt(running(fra1gt_6),454).
orientation(fra1gt_3,90,455). coords(fra1gt_3,104,213,455).  visible(fra1gt_3,455). happensAt(active(fra1gt_3),455). orientation(fra1gt_6,166,455). coords(fra1gt_6,42,182,455).  visible(fra1gt_6,455). happensAt(running(fra1gt_6),455).
orientation(fra1gt_3,90,456). coords(fra1gt_3,104,213,456).  visible(fra1gt_3,456). happensAt(active(fra1gt_3),456). orientation(fra1gt_6,166,456). coords(fra1gt_6,41,183,456).  visible(fra1gt_6,456). happensAt(running(fra1gt_6),456).
orientation(fra1gt_3,175,457). coords(fra1gt_3,103,212,457).  visible(fra1gt_3,457). happensAt(active(fra1gt_3),457). orientation(fra1gt_6,166,457). coords(fra1gt_6,40,185,457).  visible(fra1gt_6,457). happensAt(running(fra1gt_6),457).
orientation(fra1gt_3,175,458). coords(fra1gt_3,103,212,458).  visible(fra1gt_3,458). happensAt(active(fra1gt_3),458). orientation(fra1gt_6,166,458). coords(fra1gt_6,39,186,458).  visible(fra1gt_6,458). happensAt(running(fra1gt_6),458).
orientation(fra1gt_3,175,459). coords(fra1gt_3,103,212,459).  visible(fra1gt_3,459). happensAt(active(fra1gt_3),459). orientation(fra1gt_6,166,459). coords(fra1gt_6,39,188,459).  visible(fra1gt_6,459). happensAt(running(fra1gt_6),459).
orientation(fra1gt_3,175,460). coords(fra1gt_3,103,212,460).  visible(fra1gt_3,460). happensAt(active(fra1gt_3),460). orientation(fra1gt_6,166,460). coords(fra1gt_6,38,189,460).  visible(fra1gt_6,460). happensAt(running(fra1gt_6),460).
orientation(fra1gt_3,175,461). coords(fra1gt_3,103,212,461).  visible(fra1gt_3,461). happensAt(active(fra1gt_3),461). orientation(fra1gt_6,166,461). coords(fra1gt_6,38,190,461).  visible(fra1gt_6,461). happensAt(running(fra1gt_6),461).
orientation(fra1gt_3,175,462). coords(fra1gt_3,103,211,462).  visible(fra1gt_3,462). happensAt(active(fra1gt_3),462). orientation(fra1gt_6,166,462). coords(fra1gt_6,38,191,462).  visible(fra1gt_6,462). happensAt(running(fra1gt_6),462).
orientation(fra1gt_3,175,463). coords(fra1gt_3,103,211,463).  visible(fra1gt_3,463). happensAt(active(fra1gt_3),463). orientation(fra1gt_6,166,463). coords(fra1gt_6,37,192,463).  visible(fra1gt_6,463). happensAt(running(fra1gt_6),463).
orientation(fra1gt_3,175,464). coords(fra1gt_3,103,210,464).  visible(fra1gt_3,464). happensAt(active(fra1gt_3),464). orientation(fra1gt_6,166,464). coords(fra1gt_6,36,193,464).  visible(fra1gt_6,464). happensAt(running(fra1gt_6),464).
orientation(fra1gt_3,175,465). coords(fra1gt_3,103,209,465).  visible(fra1gt_3,465). happensAt(active(fra1gt_3),465). orientation(fra1gt_6,166,465). coords(fra1gt_6,36,194,465).  visible(fra1gt_6,465). happensAt(running(fra1gt_6),465).
orientation(fra1gt_3,175,466). coords(fra1gt_3,103,209,466).  visible(fra1gt_3,466). happensAt(active(fra1gt_3),466). orientation(fra1gt_6,166,466). coords(fra1gt_6,36,195,466).  visible(fra1gt_6,466). happensAt(running(fra1gt_6),466).
orientation(fra1gt_3,175,467). coords(fra1gt_3,103,208,467).  visible(fra1gt_3,467). happensAt(active(fra1gt_3),467). orientation(fra1gt_6,166,467). coords(fra1gt_6,35,196,467).  visible(fra1gt_6,467). happensAt(running(fra1gt_6),467).
orientation(fra1gt_3,175,468). coords(fra1gt_3,103,207,468).  visible(fra1gt_3,468). happensAt(active(fra1gt_3),468). orientation(fra1gt_6,166,468). coords(fra1gt_6,35,196,468).  visible(fra1gt_6,468). happensAt(running(fra1gt_6),468).
orientation(fra1gt_3,175,469). coords(fra1gt_3,103,207,469).  visible(fra1gt_3,469). happensAt(active(fra1gt_3),469). orientation(fra1gt_6,166,469). coords(fra1gt_6,35,197,469).  visible(fra1gt_6,469). happensAt(running(fra1gt_6),469).
orientation(fra1gt_3,175,470). coords(fra1gt_3,103,207,470).  visible(fra1gt_3,470). happensAt(active(fra1gt_3),470). orientation(fra1gt_6,166,470). coords(fra1gt_6,35,199,470).  visible(fra1gt_6,470). happensAt(running(fra1gt_6),470).
orientation(fra1gt_3,175,471). coords(fra1gt_3,103,207,471).  visible(fra1gt_3,471). happensAt(active(fra1gt_3),471). orientation(fra1gt_6,166,471). coords(fra1gt_6,34,199,471).  visible(fra1gt_6,471). happensAt(running(fra1gt_6),471).
orientation(fra1gt_3,175,472). coords(fra1gt_3,103,207,472).  visible(fra1gt_3,472). happensAt(active(fra1gt_3),472). orientation(fra1gt_6,166,472). coords(fra1gt_6,34,200,472).  visible(fra1gt_6,472). happensAt(running(fra1gt_6),472).
orientation(fra1gt_3,175,473). coords(fra1gt_3,103,207,473).  visible(fra1gt_3,473). happensAt(active(fra1gt_3),473). orientation(fra1gt_6,166,473). coords(fra1gt_6,34,202,473).  visible(fra1gt_6,473). happensAt(running(fra1gt_6),473).
orientation(fra1gt_3,175,474). coords(fra1gt_3,103,207,474).  visible(fra1gt_3,474). happensAt(active(fra1gt_3),474). orientation(fra1gt_6,166,474). coords(fra1gt_6,32,203,474).  visible(fra1gt_6,474). happensAt(running(fra1gt_6),474).
orientation(fra1gt_3,175,475). coords(fra1gt_3,103,206,475).  visible(fra1gt_3,475). happensAt(active(fra1gt_3),475). orientation(fra1gt_6,166,475). coords(fra1gt_6,32,204,475).  visible(fra1gt_6,475). happensAt(running(fra1gt_6),475).
orientation(fra1gt_3,175,476). coords(fra1gt_3,102,205,476).  visible(fra1gt_3,476). happensAt(active(fra1gt_3),476). orientation(fra1gt_6,166,476). coords(fra1gt_6,32,204,476).  visible(fra1gt_6,476). happensAt(running(fra1gt_6),476).
orientation(fra1gt_3,175,477). coords(fra1gt_3,102,205,477).  visible(fra1gt_3,477). happensAt(active(fra1gt_3),477). orientation(fra1gt_6,166,477). coords(fra1gt_6,32,204,477).  visible(fra1gt_6,477). happensAt(running(fra1gt_6),477).
#show holdsAt/2.