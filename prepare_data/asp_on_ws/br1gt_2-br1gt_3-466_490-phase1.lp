
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
orientation(br1gt_2,12,479). coords(br1gt_2,86,208,479).  visible(br1gt_2,479). happensAt(inactive(br1gt_2),479). orientation(br1gt_3,160,479). coords(br1gt_3,90,141,479).  visible(br1gt_3,479). happensAt(walking(br1gt_3),479).
orientation(br1gt_2,12,480). coords(br1gt_2,86,208,480).  visible(br1gt_2,480). happensAt(inactive(br1gt_2),480). orientation(br1gt_3,160,480). coords(br1gt_3,90,141,480).  visible(br1gt_3,480). happensAt(walking(br1gt_3),480).
orientation(br1gt_2,12,481). coords(br1gt_2,86,208,481).  visible(br1gt_2,481). happensAt(inactive(br1gt_2),481). orientation(br1gt_3,160,481). coords(br1gt_3,90,141,481).  visible(br1gt_3,481). happensAt(walking(br1gt_3),481).
orientation(br1gt_2,12,482). coords(br1gt_2,86,208,482).  visible(br1gt_2,482). happensAt(inactive(br1gt_2),482). orientation(br1gt_3,160,482). coords(br1gt_3,91,141,482).  visible(br1gt_3,482). happensAt(walking(br1gt_3),482).
orientation(br1gt_2,12,483). coords(br1gt_2,86,208,483).  visible(br1gt_2,483). happensAt(inactive(br1gt_2),483). orientation(br1gt_3,160,483). coords(br1gt_3,91,140,483).  visible(br1gt_3,483). happensAt(walking(br1gt_3),483).
orientation(br1gt_2,12,484). coords(br1gt_2,86,208,484).  visible(br1gt_2,484). happensAt(inactive(br1gt_2),484). orientation(br1gt_3,160,484). coords(br1gt_3,91,140,484).  visible(br1gt_3,484). happensAt(walking(br1gt_3),484).
orientation(br1gt_2,12,485). coords(br1gt_2,86,208,485).  visible(br1gt_2,485). happensAt(inactive(br1gt_2),485). orientation(br1gt_3,160,485). coords(br1gt_3,93,140,485).  visible(br1gt_3,485). happensAt(walking(br1gt_3),485).
orientation(br1gt_2,12,486). coords(br1gt_2,86,208,486).  visible(br1gt_2,486). happensAt(inactive(br1gt_2),486). orientation(br1gt_3,160,486). coords(br1gt_3,94,140,486).  visible(br1gt_3,486). happensAt(walking(br1gt_3),486).
orientation(br1gt_2,12,487). coords(br1gt_2,86,209,487).  visible(br1gt_2,487). happensAt(inactive(br1gt_2),487). orientation(br1gt_3,160,487). coords(br1gt_3,96,140,487).  visible(br1gt_3,487). happensAt(walking(br1gt_3),487).
orientation(br1gt_2,12,488). coords(br1gt_2,86,209,488).  visible(br1gt_2,488). happensAt(inactive(br1gt_2),488). orientation(br1gt_3,160,488). coords(br1gt_3,97,140,488).  visible(br1gt_3,488). happensAt(walking(br1gt_3),488).
orientation(br1gt_2,12,489). coords(br1gt_2,86,209,489).  visible(br1gt_2,489). happensAt(inactive(br1gt_2),489). orientation(br1gt_3,160,489). coords(br1gt_3,98,140,489).  visible(br1gt_3,489). happensAt(walking(br1gt_3),489).
orientation(br1gt_2,12,490). coords(br1gt_2,86,209,490).  visible(br1gt_2,490). happensAt(inactive(br1gt_2),490). orientation(br1gt_3,160,490). coords(br1gt_3,99,140,490).  visible(br1gt_3,490). happensAt(walking(br1gt_3),490).
#show holdsAt/2.