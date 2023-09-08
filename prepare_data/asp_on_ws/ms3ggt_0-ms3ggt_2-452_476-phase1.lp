
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

	orientation(ms3ggt_0,166,452). coords(ms3ggt_0,85,150,452).  visible(ms3ggt_0,452). happensAt(walking(ms3ggt_0),452). orientation(ms3ggt_2,155,452). coords(ms3ggt_2,106,160,452).  visible(ms3ggt_2,452). happensAt(walking(ms3ggt_2),452).
orientation(ms3ggt_0,166,453). coords(ms3ggt_0,85,150,453).  visible(ms3ggt_0,453). happensAt(walking(ms3ggt_0),453). orientation(ms3ggt_2,156,453). coords(ms3ggt_2,106,159,453).  visible(ms3ggt_2,453). happensAt(walking(ms3ggt_2),453).
orientation(ms3ggt_0,166,454). coords(ms3ggt_0,83,151,454).  visible(ms3ggt_0,454). happensAt(walking(ms3ggt_0),454). orientation(ms3ggt_2,156,454). coords(ms3ggt_2,105,159,454).  visible(ms3ggt_2,454). happensAt(walking(ms3ggt_2),454).
orientation(ms3ggt_0,166,455). coords(ms3ggt_0,82,151,455).  visible(ms3ggt_0,455). happensAt(walking(ms3ggt_0),455). orientation(ms3ggt_2,156,455). coords(ms3ggt_2,105,158,455).  visible(ms3ggt_2,455). happensAt(walking(ms3ggt_2),455).
orientation(ms3ggt_0,166,456). coords(ms3ggt_0,82,151,456).  visible(ms3ggt_0,456). happensAt(walking(ms3ggt_0),456). orientation(ms3ggt_2,156,456). coords(ms3ggt_2,104,157,456).  visible(ms3ggt_2,456). happensAt(walking(ms3ggt_2),456).
orientation(ms3ggt_0,166,457). coords(ms3ggt_0,78,152,457).  visible(ms3ggt_0,457). happensAt(walking(ms3ggt_0),457). orientation(ms3ggt_2,156,457). coords(ms3ggt_2,104,158,457).  visible(ms3ggt_2,457). happensAt(walking(ms3ggt_2),457).
orientation(ms3ggt_0,166,458). coords(ms3ggt_0,77,152,458).  visible(ms3ggt_0,458). happensAt(walking(ms3ggt_0),458). orientation(ms3ggt_2,156,458). coords(ms3ggt_2,103,156,458).  visible(ms3ggt_2,458). happensAt(walking(ms3ggt_2),458).
orientation(ms3ggt_0,166,459). coords(ms3ggt_0,86,156,459).  visible(ms3ggt_0,459). happensAt(walking(ms3ggt_0),459). orientation(ms3ggt_2,156,459). coords(ms3ggt_2,103,156,459).  visible(ms3ggt_2,459). happensAt(walking(ms3ggt_2),459).
orientation(ms3ggt_0,166,460). coords(ms3ggt_0,86,157,460).  visible(ms3ggt_0,460). happensAt(walking(ms3ggt_0),460). orientation(ms3ggt_2,156,460). coords(ms3ggt_2,103,155,460).  visible(ms3ggt_2,460). happensAt(walking(ms3ggt_2),460).
orientation(ms3ggt_0,166,461). coords(ms3ggt_0,87,158,461).  visible(ms3ggt_0,461). happensAt(walking(ms3ggt_0),461). orientation(ms3ggt_2,156,461). coords(ms3ggt_2,103,155,461).  visible(ms3ggt_2,461). happensAt(walking(ms3ggt_2),461).
orientation(ms3ggt_0,166,462). coords(ms3ggt_0,86,158,462).  visible(ms3ggt_0,462). happensAt(walking(ms3ggt_0),462). orientation(ms3ggt_2,151,462). coords(ms3ggt_2,103,154,462).  visible(ms3ggt_2,462). happensAt(walking(ms3ggt_2),462).
orientation(ms3ggt_0,166,463). coords(ms3ggt_0,85,161,463).  visible(ms3ggt_0,463). happensAt(walking(ms3ggt_0),463). orientation(ms3ggt_2,151,463). coords(ms3ggt_2,103,153,463).  visible(ms3ggt_2,463). happensAt(walking(ms3ggt_2),463).
orientation(ms3ggt_0,155,464). coords(ms3ggt_0,84,162,464).  visible(ms3ggt_0,464). happensAt(walking(ms3ggt_0),464). orientation(ms3ggt_2,151,464). coords(ms3ggt_2,102,152,464).  visible(ms3ggt_2,464). happensAt(walking(ms3ggt_2),464).
orientation(ms3ggt_0,155,465). coords(ms3ggt_0,85,163,465).  visible(ms3ggt_0,465). happensAt(walking(ms3ggt_0),465). orientation(ms3ggt_2,151,465). coords(ms3ggt_2,101,152,465).  visible(ms3ggt_2,465). happensAt(walking(ms3ggt_2),465).
orientation(ms3ggt_0,160,466). coords(ms3ggt_0,85,166,466).  visible(ms3ggt_0,466). happensAt(walking(ms3ggt_0),466). orientation(ms3ggt_2,151,466). coords(ms3ggt_2,101,150,466).  visible(ms3ggt_2,466). happensAt(walking(ms3ggt_2),466).
orientation(ms3ggt_0,151,467). coords(ms3ggt_0,85,167,467).  visible(ms3ggt_0,467). happensAt(walking(ms3ggt_0),467). orientation(ms3ggt_2,151,467). coords(ms3ggt_2,100,148,467).  visible(ms3ggt_2,467). happensAt(walking(ms3ggt_2),467).
orientation(ms3ggt_0,151,468). coords(ms3ggt_0,84,168,468).  visible(ms3ggt_0,468). happensAt(walking(ms3ggt_0),468). orientation(ms3ggt_2,145,468). coords(ms3ggt_2,99,147,468).  visible(ms3ggt_2,468). happensAt(walking(ms3ggt_2),468).
orientation(ms3ggt_0,151,469). coords(ms3ggt_0,84,168,469).  visible(ms3ggt_0,469). happensAt(walking(ms3ggt_0),469). orientation(ms3ggt_2,145,469). coords(ms3ggt_2,99,145,469).  visible(ms3ggt_2,469). happensAt(walking(ms3ggt_2),469).
orientation(ms3ggt_0,151,470). coords(ms3ggt_0,84,169,470).  visible(ms3ggt_0,470). happensAt(walking(ms3ggt_0),470). orientation(ms3ggt_2,145,470). coords(ms3ggt_2,99,145,470).  visible(ms3ggt_2,470). happensAt(walking(ms3ggt_2),470).
orientation(ms3ggt_0,151,471). coords(ms3ggt_0,84,169,471).  visible(ms3ggt_0,471). happensAt(walking(ms3ggt_0),471). orientation(ms3ggt_2,145,471). coords(ms3ggt_2,99,145,471).  visible(ms3ggt_2,471). happensAt(walking(ms3ggt_2),471).
orientation(ms3ggt_0,151,472). coords(ms3ggt_0,83,170,472).  visible(ms3ggt_0,472). happensAt(walking(ms3ggt_0),472). orientation(ms3ggt_2,145,472). coords(ms3ggt_2,99,143,472).  visible(ms3ggt_2,472). happensAt(walking(ms3ggt_2),472).
orientation(ms3ggt_0,151,473). coords(ms3ggt_0,81,170,473).  visible(ms3ggt_0,473). happensAt(walking(ms3ggt_0),473). orientation(ms3ggt_2,158,473). coords(ms3ggt_2,99,142,473).  visible(ms3ggt_2,473). happensAt(walking(ms3ggt_2),473).
orientation(ms3ggt_0,151,474). coords(ms3ggt_0,80,171,474).  visible(ms3ggt_0,474). happensAt(walking(ms3ggt_0),474). orientation(ms3ggt_2,158,474). coords(ms3ggt_2,99,142,474).  visible(ms3ggt_2,474). happensAt(walking(ms3ggt_2),474).
orientation(ms3ggt_0,151,475). coords(ms3ggt_0,79,172,475).  visible(ms3ggt_0,475). happensAt(walking(ms3ggt_0),475). orientation(ms3ggt_2,158,475). coords(ms3ggt_2,99,141,475).  visible(ms3ggt_2,475). happensAt(walking(ms3ggt_2),475).
orientation(ms3ggt_0,151,476). coords(ms3ggt_0,79,172,476).  visible(ms3ggt_0,476). happensAt(walking(ms3ggt_0),476). orientation(ms3ggt_2,158,476). coords(ms3ggt_2,99,141,476).  visible(ms3ggt_2,476). happensAt(walking(ms3ggt_2),476).
#show holdsAt/2.