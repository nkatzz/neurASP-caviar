
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

	orientation(bww1gt_1,0,458). coords(bww1gt_1,105,264,458).  visible(bww1gt_1,458). happensAt(inactive(bww1gt_1),458). orientation(bww1gt_2,151,458). coords(bww1gt_2,162,148,458).  visible(bww1gt_2,458). happensAt(walking(bww1gt_2),458).
orientation(bww1gt_1,0,459). coords(bww1gt_1,105,264,459).  visible(bww1gt_1,459). happensAt(inactive(bww1gt_1),459). orientation(bww1gt_2,150,459). coords(bww1gt_2,162,148,459).  visible(bww1gt_2,459). happensAt(walking(bww1gt_2),459).
orientation(bww1gt_1,0,460). coords(bww1gt_1,105,264,460).  visible(bww1gt_1,460). happensAt(inactive(bww1gt_1),460). orientation(bww1gt_2,150,460). coords(bww1gt_2,163,148,460).  visible(bww1gt_2,460). happensAt(walking(bww1gt_2),460).
orientation(bww1gt_1,0,461). coords(bww1gt_1,105,264,461).  visible(bww1gt_1,461). happensAt(inactive(bww1gt_1),461). orientation(bww1gt_2,148,461). coords(bww1gt_2,163,149,461).  visible(bww1gt_2,461). happensAt(walking(bww1gt_2),461).
orientation(bww1gt_1,0,462). coords(bww1gt_1,105,264,462).  visible(bww1gt_1,462). happensAt(inactive(bww1gt_1),462). orientation(bww1gt_2,147,462). coords(bww1gt_2,164,150,462).  visible(bww1gt_2,462). happensAt(walking(bww1gt_2),462).
orientation(bww1gt_1,0,463). coords(bww1gt_1,105,264,463).  visible(bww1gt_1,463). happensAt(inactive(bww1gt_1),463). orientation(bww1gt_2,146,463). coords(bww1gt_2,164,151,463).  visible(bww1gt_2,463). happensAt(walking(bww1gt_2),463).
orientation(bww1gt_1,0,464). coords(bww1gt_1,105,264,464).  visible(bww1gt_1,464). happensAt(inactive(bww1gt_1),464). orientation(bww1gt_2,143,464). coords(bww1gt_2,164,152,464).  visible(bww1gt_2,464). happensAt(walking(bww1gt_2),464).
orientation(bww1gt_1,0,465). coords(bww1gt_1,105,264,465).  visible(bww1gt_1,465). happensAt(inactive(bww1gt_1),465). orientation(bww1gt_2,145,465). coords(bww1gt_2,165,154,465).  visible(bww1gt_2,465). happensAt(walking(bww1gt_2),465).
orientation(bww1gt_1,0,466). coords(bww1gt_1,105,264,466).  visible(bww1gt_1,466). happensAt(inactive(bww1gt_1),466). orientation(bww1gt_2,150,466). coords(bww1gt_2,165,155,466).  visible(bww1gt_2,466). happensAt(walking(bww1gt_2),466).
orientation(bww1gt_1,0,467). coords(bww1gt_1,105,264,467).  visible(bww1gt_1,467). happensAt(inactive(bww1gt_1),467). orientation(bww1gt_2,151,467). coords(bww1gt_2,166,157,467).  visible(bww1gt_2,467). happensAt(walking(bww1gt_2),467).
orientation(bww1gt_1,0,468). coords(bww1gt_1,105,264,468).  visible(bww1gt_1,468). happensAt(inactive(bww1gt_1),468). orientation(bww1gt_2,151,468). coords(bww1gt_2,166,158,468).  visible(bww1gt_2,468). happensAt(walking(bww1gt_2),468).
orientation(bww1gt_1,0,469). coords(bww1gt_1,105,264,469).  visible(bww1gt_1,469). happensAt(inactive(bww1gt_1),469). orientation(bww1gt_2,151,469). coords(bww1gt_2,167,160,469).  visible(bww1gt_2,469). happensAt(walking(bww1gt_2),469).
orientation(bww1gt_1,0,470). coords(bww1gt_1,105,264,470).  visible(bww1gt_1,470). happensAt(inactive(bww1gt_1),470). orientation(bww1gt_2,151,470). coords(bww1gt_2,167,160,470).  visible(bww1gt_2,470). happensAt(walking(bww1gt_2),470).
orientation(bww1gt_1,0,471). coords(bww1gt_1,105,264,471).  visible(bww1gt_1,471). happensAt(inactive(bww1gt_1),471). orientation(bww1gt_2,151,471). coords(bww1gt_2,168,161,471).  visible(bww1gt_2,471). happensAt(walking(bww1gt_2),471).
orientation(bww1gt_1,0,472). coords(bww1gt_1,105,264,472).  visible(bww1gt_1,472). happensAt(inactive(bww1gt_1),472). orientation(bww1gt_2,151,472). coords(bww1gt_2,168,161,472).  visible(bww1gt_2,472). happensAt(walking(bww1gt_2),472).
orientation(bww1gt_1,0,473). coords(bww1gt_1,105,264,473).  visible(bww1gt_1,473). happensAt(inactive(bww1gt_1),473). orientation(bww1gt_2,151,473). coords(bww1gt_2,168,162,473).  visible(bww1gt_2,473). happensAt(walking(bww1gt_2),473).
orientation(bww1gt_1,0,474). coords(bww1gt_1,105,264,474).  visible(bww1gt_1,474). happensAt(inactive(bww1gt_1),474). orientation(bww1gt_2,151,474). coords(bww1gt_2,168,163,474).  visible(bww1gt_2,474). happensAt(walking(bww1gt_2),474).
orientation(bww1gt_1,0,475). coords(bww1gt_1,105,264,475).  visible(bww1gt_1,475). happensAt(inactive(bww1gt_1),475). orientation(bww1gt_2,151,475). coords(bww1gt_2,168,163,475).  visible(bww1gt_2,475). happensAt(walking(bww1gt_2),475).
orientation(bww1gt_1,0,476). coords(bww1gt_1,105,264,476).  visible(bww1gt_1,476). happensAt(inactive(bww1gt_1),476). orientation(bww1gt_2,151,476). coords(bww1gt_2,168,164,476).  visible(bww1gt_2,476). happensAt(walking(bww1gt_2),476).
orientation(bww1gt_1,0,477). coords(bww1gt_1,105,264,477).  visible(bww1gt_1,477). happensAt(inactive(bww1gt_1),477). orientation(bww1gt_2,151,477). coords(bww1gt_2,168,164,477).  visible(bww1gt_2,477). happensAt(walking(bww1gt_2),477).
orientation(bww1gt_1,0,478). coords(bww1gt_1,105,264,478).  visible(bww1gt_1,478). happensAt(inactive(bww1gt_1),478). orientation(bww1gt_2,151,478). coords(bww1gt_2,169,165,478).  visible(bww1gt_2,478). happensAt(walking(bww1gt_2),478).
orientation(bww1gt_1,0,479). coords(bww1gt_1,105,264,479).  visible(bww1gt_1,479). happensAt(inactive(bww1gt_1),479). orientation(bww1gt_2,151,479). coords(bww1gt_2,169,165,479).  visible(bww1gt_2,479). happensAt(walking(bww1gt_2),479).
orientation(bww1gt_1,0,480). coords(bww1gt_1,105,264,480).  visible(bww1gt_1,480). happensAt(inactive(bww1gt_1),480). orientation(bww1gt_2,151,480). coords(bww1gt_2,170,165,480).  visible(bww1gt_2,480). happensAt(walking(bww1gt_2),480).
orientation(bww1gt_1,0,481). coords(bww1gt_1,105,264,481).  visible(bww1gt_1,481). happensAt(inactive(bww1gt_1),481). orientation(bww1gt_2,169,481). coords(bww1gt_2,171,166,481).  visible(bww1gt_2,481). happensAt(walking(bww1gt_2),481).
orientation(bww1gt_1,0,482). coords(bww1gt_1,105,264,482).  visible(bww1gt_1,482). happensAt(inactive(bww1gt_1),482). orientation(bww1gt_2,156,482). coords(bww1gt_2,172,166,482).  visible(bww1gt_2,482). happensAt(walking(bww1gt_2),482).
#show holdsAt/2.