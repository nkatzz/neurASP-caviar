
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
orientation(ms3ggt_0,151,471). coords(ms3ggt_0,84,169,471).  visible(ms3ggt_0,471). happensAt(walking(ms3ggt_0),471). orientation(ms3ggt_1,174,471). coords(ms3ggt_1,235,211,471).  visible(ms3ggt_1,471). happensAt(walking(ms3ggt_1),471).
orientation(ms3ggt_0,151,472). coords(ms3ggt_0,83,170,472).  visible(ms3ggt_0,472). happensAt(walking(ms3ggt_0),472). orientation(ms3ggt_1,174,472). coords(ms3ggt_1,237,213,472).  visible(ms3ggt_1,472). happensAt(walking(ms3ggt_1),472).
orientation(ms3ggt_0,151,473). coords(ms3ggt_0,81,170,473).  visible(ms3ggt_0,473). happensAt(walking(ms3ggt_0),473). orientation(ms3ggt_1,177,473). coords(ms3ggt_1,240,216,473).  visible(ms3ggt_1,473). happensAt(walking(ms3ggt_1),473).
orientation(ms3ggt_0,151,474). coords(ms3ggt_0,80,171,474).  visible(ms3ggt_0,474). happensAt(walking(ms3ggt_0),474). orientation(ms3ggt_1,177,474). coords(ms3ggt_1,243,218,474).  visible(ms3ggt_1,474). happensAt(walking(ms3ggt_1),474).
orientation(ms3ggt_0,151,475). coords(ms3ggt_0,79,172,475).  visible(ms3ggt_0,475). happensAt(walking(ms3ggt_0),475). orientation(ms3ggt_1,177,475). coords(ms3ggt_1,244,220,475).  visible(ms3ggt_1,475). happensAt(walking(ms3ggt_1),475).
orientation(ms3ggt_0,151,476). coords(ms3ggt_0,79,172,476).  visible(ms3ggt_0,476). happensAt(walking(ms3ggt_0),476). orientation(ms3ggt_1,177,476). coords(ms3ggt_1,246,221,476).  visible(ms3ggt_1,476). happensAt(walking(ms3ggt_1),476).
orientation(ms3ggt_0,151,477). coords(ms3ggt_0,79,172,477).  visible(ms3ggt_0,477). happensAt(walking(ms3ggt_0),477). orientation(ms3ggt_1,177,477). coords(ms3ggt_1,246,222,477).  visible(ms3ggt_1,477). happensAt(walking(ms3ggt_1),477).
orientation(ms3ggt_0,151,478). coords(ms3ggt_0,79,172,478).  visible(ms3ggt_0,478). happensAt(walking(ms3ggt_0),478). orientation(ms3ggt_1,177,478). coords(ms3ggt_1,247,223,478).  visible(ms3ggt_1,478). happensAt(walking(ms3ggt_1),478).
orientation(ms3ggt_0,151,479). coords(ms3ggt_0,79,172,479).  visible(ms3ggt_0,479). happensAt(walking(ms3ggt_0),479). orientation(ms3ggt_1,5,479). coords(ms3ggt_1,248,224,479).  visible(ms3ggt_1,479). happensAt(walking(ms3ggt_1),479).
orientation(ms3ggt_0,151,480). coords(ms3ggt_0,79,172,480).  visible(ms3ggt_0,480). happensAt(walking(ms3ggt_0),480). orientation(ms3ggt_1,14,480). coords(ms3ggt_1,248,224,480).  visible(ms3ggt_1,480). happensAt(walking(ms3ggt_1),480).
orientation(ms3ggt_0,156,481). coords(ms3ggt_0,78,178,481).  visible(ms3ggt_0,481). happensAt(walking(ms3ggt_0),481). orientation(ms3ggt_1,14,481). coords(ms3ggt_1,249,226,481).  visible(ms3ggt_1,481). happensAt(walking(ms3ggt_1),481).
orientation(ms3ggt_0,156,482). coords(ms3ggt_0,78,179,482).  visible(ms3ggt_0,482). happensAt(walking(ms3ggt_0),482). orientation(ms3ggt_1,14,482). coords(ms3ggt_1,249,228,482).  visible(ms3ggt_1,482). happensAt(walking(ms3ggt_1),482).
#show holdsAt/2.