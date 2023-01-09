
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

	orientation(wk1gt_4,147,459). coords(wk1gt_4,56,169,459).  visible(wk1gt_4,459). happensAt(active(wk1gt_4),459). orientation(wk1gt_5,162,459). coords(wk1gt_5,55,156,459).  visible(wk1gt_5,459). happensAt(walking(wk1gt_5),459).
orientation(wk1gt_4,147,460). coords(wk1gt_4,56,169,460).  visible(wk1gt_4,460). happensAt(active(wk1gt_4),460). orientation(wk1gt_5,162,460). coords(wk1gt_5,55,155,460).  visible(wk1gt_5,460). happensAt(walking(wk1gt_5),460).
orientation(wk1gt_4,147,461). coords(wk1gt_4,56,169,461).  visible(wk1gt_4,461). happensAt(active(wk1gt_4),461). orientation(wk1gt_5,162,461). coords(wk1gt_5,55,155,461).  visible(wk1gt_5,461). happensAt(walking(wk1gt_5),461).
orientation(wk1gt_4,147,462). coords(wk1gt_4,56,169,462).  visible(wk1gt_4,462). happensAt(active(wk1gt_4),462). orientation(wk1gt_5,162,462). coords(wk1gt_5,56,154,462).  visible(wk1gt_5,462). happensAt(walking(wk1gt_5),462).
orientation(wk1gt_4,147,463). coords(wk1gt_4,56,169,463).  visible(wk1gt_4,463). happensAt(active(wk1gt_4),463). orientation(wk1gt_5,162,463). coords(wk1gt_5,56,154,463).  visible(wk1gt_5,463). happensAt(walking(wk1gt_5),463).
orientation(wk1gt_4,147,464). coords(wk1gt_4,56,169,464).  visible(wk1gt_4,464). happensAt(active(wk1gt_4),464). orientation(wk1gt_5,162,464). coords(wk1gt_5,56,154,464).  visible(wk1gt_5,464). happensAt(walking(wk1gt_5),464).
orientation(wk1gt_4,147,465). coords(wk1gt_4,56,169,465).  visible(wk1gt_4,465). happensAt(active(wk1gt_4),465). orientation(wk1gt_5,162,465). coords(wk1gt_5,56,153,465).  visible(wk1gt_5,465). happensAt(walking(wk1gt_5),465).
orientation(wk1gt_4,147,466). coords(wk1gt_4,56,169,466).  visible(wk1gt_4,466). happensAt(active(wk1gt_4),466). orientation(wk1gt_5,162,466). coords(wk1gt_5,56,153,466).  visible(wk1gt_5,466). happensAt(walking(wk1gt_5),466).
orientation(wk1gt_4,147,467). coords(wk1gt_4,56,169,467).  visible(wk1gt_4,467). happensAt(active(wk1gt_4),467). orientation(wk1gt_5,162,467). coords(wk1gt_5,56,153,467).  visible(wk1gt_5,467). happensAt(walking(wk1gt_5),467).
orientation(wk1gt_4,147,468). coords(wk1gt_4,56,169,468).  visible(wk1gt_4,468). happensAt(active(wk1gt_4),468). orientation(wk1gt_5,162,468). coords(wk1gt_5,56,153,468).  visible(wk1gt_5,468). happensAt(walking(wk1gt_5),468).
orientation(wk1gt_4,147,469). coords(wk1gt_4,56,168,469).  visible(wk1gt_4,469). happensAt(active(wk1gt_4),469). orientation(wk1gt_5,162,469). coords(wk1gt_5,56,153,469).  visible(wk1gt_5,469). happensAt(walking(wk1gt_5),469).
orientation(wk1gt_4,147,470). coords(wk1gt_4,56,168,470).  visible(wk1gt_4,470). happensAt(active(wk1gt_4),470). orientation(wk1gt_5,162,470). coords(wk1gt_5,56,153,470).  visible(wk1gt_5,470). happensAt(walking(wk1gt_5),470).
orientation(wk1gt_4,147,471). coords(wk1gt_4,56,168,471).  visible(wk1gt_4,471). happensAt(active(wk1gt_4),471). orientation(wk1gt_5,162,471). coords(wk1gt_5,57,153,471).  visible(wk1gt_5,471). happensAt(walking(wk1gt_5),471).
orientation(wk1gt_4,147,472). coords(wk1gt_4,56,168,472).  visible(wk1gt_4,472). happensAt(active(wk1gt_4),472). orientation(wk1gt_5,162,472). coords(wk1gt_5,57,153,472).  visible(wk1gt_5,472). happensAt(walking(wk1gt_5),472).
orientation(wk1gt_4,147,473). coords(wk1gt_4,56,168,473).  visible(wk1gt_4,473). happensAt(active(wk1gt_4),473). orientation(wk1gt_5,162,473). coords(wk1gt_5,57,153,473).  visible(wk1gt_5,473). happensAt(walking(wk1gt_5),473).
orientation(wk1gt_4,147,474). coords(wk1gt_4,56,168,474).  visible(wk1gt_4,474). happensAt(active(wk1gt_4),474). orientation(wk1gt_5,162,474). coords(wk1gt_5,57,153,474).  visible(wk1gt_5,474). happensAt(walking(wk1gt_5),474).
orientation(wk1gt_4,147,475). coords(wk1gt_4,56,168,475).  visible(wk1gt_4,475). happensAt(active(wk1gt_4),475). orientation(wk1gt_5,162,475). coords(wk1gt_5,57,153,475).  visible(wk1gt_5,475). happensAt(walking(wk1gt_5),475).
orientation(wk1gt_4,147,476). coords(wk1gt_4,56,168,476).  visible(wk1gt_4,476). happensAt(active(wk1gt_4),476). orientation(wk1gt_5,162,476). coords(wk1gt_5,57,152,476).  visible(wk1gt_5,476). happensAt(active(wk1gt_5),476).
orientation(wk1gt_4,147,477). coords(wk1gt_4,56,168,477).  visible(wk1gt_4,477). happensAt(active(wk1gt_4),477). orientation(wk1gt_5,162,477). coords(wk1gt_5,57,152,477).  visible(wk1gt_5,477). happensAt(active(wk1gt_5),477).
orientation(wk1gt_4,147,478). coords(wk1gt_4,56,168,478).  visible(wk1gt_4,478). happensAt(active(wk1gt_4),478). orientation(wk1gt_5,162,478). coords(wk1gt_5,57,152,478).  visible(wk1gt_5,478). happensAt(active(wk1gt_5),478).
orientation(wk1gt_4,147,479). coords(wk1gt_4,56,168,479).  visible(wk1gt_4,479). happensAt(active(wk1gt_4),479). orientation(wk1gt_5,162,479). coords(wk1gt_5,57,152,479).  visible(wk1gt_5,479). happensAt(active(wk1gt_5),479).
orientation(wk1gt_4,147,480). coords(wk1gt_4,56,169,480).  visible(wk1gt_4,480). happensAt(active(wk1gt_4),480). orientation(wk1gt_5,162,480). coords(wk1gt_5,57,152,480).  visible(wk1gt_5,480). happensAt(active(wk1gt_5),480).
orientation(wk1gt_4,147,481). coords(wk1gt_4,56,169,481).  visible(wk1gt_4,481). happensAt(active(wk1gt_4),481). orientation(wk1gt_5,162,481). coords(wk1gt_5,57,152,481).  visible(wk1gt_5,481). happensAt(active(wk1gt_5),481).
orientation(wk1gt_4,147,482). coords(wk1gt_4,56,169,482).  visible(wk1gt_4,482). happensAt(active(wk1gt_4),482). orientation(wk1gt_5,162,482). coords(wk1gt_5,57,152,482).  visible(wk1gt_5,482). happensAt(active(wk1gt_5),482).
orientation(wk1gt_4,147,483). coords(wk1gt_4,56,168,483).  visible(wk1gt_4,483). happensAt(active(wk1gt_4),483). orientation(wk1gt_5,162,483). coords(wk1gt_5,57,152,483).  visible(wk1gt_5,483). happensAt(active(wk1gt_5),483).
#show holdsAt/2.