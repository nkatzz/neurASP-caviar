
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
orientation(wk1gt_1,147,477). coords(wk1gt_1,81,78,477).  visible(wk1gt_1,477). happensAt(walking(wk1gt_1),477). orientation(wk1gt_5,162,477). coords(wk1gt_5,57,152,477).  visible(wk1gt_5,477). happensAt(active(wk1gt_5),477).
orientation(wk1gt_1,147,478). coords(wk1gt_1,81,78,478).  visible(wk1gt_1,478). happensAt(walking(wk1gt_1),478). orientation(wk1gt_5,162,478). coords(wk1gt_5,57,152,478).  visible(wk1gt_5,478). happensAt(active(wk1gt_5),478).
orientation(wk1gt_1,147,479). coords(wk1gt_1,81,78,479).  visible(wk1gt_1,479). happensAt(walking(wk1gt_1),479). orientation(wk1gt_5,162,479). coords(wk1gt_5,57,152,479).  visible(wk1gt_5,479). happensAt(active(wk1gt_5),479).
orientation(wk1gt_1,140,480). coords(wk1gt_1,81,78,480).  visible(wk1gt_1,480). happensAt(walking(wk1gt_1),480). orientation(wk1gt_5,162,480). coords(wk1gt_5,57,152,480).  visible(wk1gt_5,480). happensAt(active(wk1gt_5),480).
orientation(wk1gt_1,140,481). coords(wk1gt_1,81,78,481).  visible(wk1gt_1,481). happensAt(walking(wk1gt_1),481). orientation(wk1gt_5,162,481). coords(wk1gt_5,57,152,481).  visible(wk1gt_5,481). happensAt(active(wk1gt_5),481).
orientation(wk1gt_1,140,482). coords(wk1gt_1,81,77,482).  visible(wk1gt_1,482). happensAt(walking(wk1gt_1),482). orientation(wk1gt_5,162,482). coords(wk1gt_5,57,152,482).  visible(wk1gt_5,482). happensAt(active(wk1gt_5),482).
orientation(wk1gt_1,140,483). coords(wk1gt_1,81,76,483).  visible(wk1gt_1,483). happensAt(walking(wk1gt_1),483). orientation(wk1gt_5,162,483). coords(wk1gt_5,57,152,483).  visible(wk1gt_5,483). happensAt(active(wk1gt_5),483).
orientation(wk1gt_1,140,484). coords(wk1gt_1,81,75,484).  visible(wk1gt_1,484). happensAt(walking(wk1gt_1),484). orientation(wk1gt_5,162,484). coords(wk1gt_5,57,152,484).  visible(wk1gt_5,484). happensAt(active(wk1gt_5),484).
orientation(wk1gt_1,140,485). coords(wk1gt_1,81,75,485).  visible(wk1gt_1,485). happensAt(walking(wk1gt_1),485). orientation(wk1gt_5,162,485). coords(wk1gt_5,57,152,485).  visible(wk1gt_5,485). happensAt(active(wk1gt_5),485).
orientation(wk1gt_1,140,486). coords(wk1gt_1,81,73,486).  visible(wk1gt_1,486). happensAt(walking(wk1gt_1),486). orientation(wk1gt_5,162,486). coords(wk1gt_5,57,152,486).  visible(wk1gt_5,486). happensAt(active(wk1gt_5),486).
orientation(wk1gt_1,140,487). coords(wk1gt_1,81,72,487).  visible(wk1gt_1,487). happensAt(walking(wk1gt_1),487). orientation(wk1gt_5,162,487). coords(wk1gt_5,57,152,487).  visible(wk1gt_5,487). happensAt(active(wk1gt_5),487).
orientation(wk1gt_1,140,488). coords(wk1gt_1,81,72,488).  visible(wk1gt_1,488). happensAt(walking(wk1gt_1),488). orientation(wk1gt_5,162,488). coords(wk1gt_5,57,152,488).  visible(wk1gt_5,488). happensAt(active(wk1gt_5),488).
#show holdsAt/2.