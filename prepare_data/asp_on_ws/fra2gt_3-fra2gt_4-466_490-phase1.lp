
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

	orientation(fra2gt_3,153,466). coords(fra2gt_3,38,161,466).  visible(fra2gt_3,466). happensAt(walking(fra2gt_3),466). orientation(fra2gt_4,150,466). coords(fra2gt_4,68,81,466).  visible(fra2gt_4,466). happensAt(walking(fra2gt_4),466).
orientation(fra2gt_3,153,467). coords(fra2gt_3,38,160,467).  visible(fra2gt_3,467). happensAt(walking(fra2gt_3),467). orientation(fra2gt_4,150,467). coords(fra2gt_4,68,81,467).  visible(fra2gt_4,467). happensAt(walking(fra2gt_4),467).
orientation(fra2gt_3,150,468). coords(fra2gt_3,38,160,468).  visible(fra2gt_3,468). happensAt(walking(fra2gt_3),468). orientation(fra2gt_4,150,468). coords(fra2gt_4,66,81,468).  visible(fra2gt_4,468). happensAt(walking(fra2gt_4),468).
orientation(fra2gt_3,150,469). coords(fra2gt_3,38,160,469).  visible(fra2gt_3,469). happensAt(walking(fra2gt_3),469). orientation(fra2gt_4,150,469). coords(fra2gt_4,66,81,469).  visible(fra2gt_4,469). happensAt(walking(fra2gt_4),469).
orientation(fra2gt_3,150,470). coords(fra2gt_3,38,159,470).  visible(fra2gt_3,470). happensAt(walking(fra2gt_3),470). orientation(fra2gt_4,150,470). coords(fra2gt_4,66,82,470).  visible(fra2gt_4,470). happensAt(walking(fra2gt_4),470).
orientation(fra2gt_3,150,471). coords(fra2gt_3,38,158,471).  visible(fra2gt_3,471). happensAt(walking(fra2gt_3),471). orientation(fra2gt_4,150,471). coords(fra2gt_4,66,81,471).  visible(fra2gt_4,471). happensAt(walking(fra2gt_4),471).
orientation(fra2gt_3,150,472). coords(fra2gt_3,38,157,472).  visible(fra2gt_3,472). happensAt(walking(fra2gt_3),472). orientation(fra2gt_4,150,472). coords(fra2gt_4,66,81,472).  visible(fra2gt_4,472). happensAt(walking(fra2gt_4),472).
orientation(fra2gt_3,150,473). coords(fra2gt_3,38,156,473).  visible(fra2gt_3,473). happensAt(walking(fra2gt_3),473). orientation(fra2gt_4,150,473). coords(fra2gt_4,65,81,473).  visible(fra2gt_4,473). happensAt(walking(fra2gt_4),473).
orientation(fra2gt_3,150,474). coords(fra2gt_3,38,155,474).  visible(fra2gt_3,474). happensAt(walking(fra2gt_3),474). orientation(fra2gt_4,150,474). coords(fra2gt_4,65,81,474).  visible(fra2gt_4,474). happensAt(walking(fra2gt_4),474).
orientation(fra2gt_3,150,475). coords(fra2gt_3,38,155,475).  visible(fra2gt_3,475). happensAt(walking(fra2gt_3),475). orientation(fra2gt_4,150,475). coords(fra2gt_4,65,81,475).  visible(fra2gt_4,475). happensAt(walking(fra2gt_4),475).
orientation(fra2gt_3,150,476). coords(fra2gt_3,38,155,476).  visible(fra2gt_3,476). happensAt(walking(fra2gt_3),476). orientation(fra2gt_4,150,476). coords(fra2gt_4,64,81,476).  visible(fra2gt_4,476). happensAt(walking(fra2gt_4),476).
orientation(fra2gt_3,150,477). coords(fra2gt_3,38,154,477).  visible(fra2gt_3,477). happensAt(walking(fra2gt_3),477). orientation(fra2gt_4,150,477). coords(fra2gt_4,64,81,477).  visible(fra2gt_4,477). happensAt(walking(fra2gt_4),477).
orientation(fra2gt_3,150,478). coords(fra2gt_3,38,154,478).  visible(fra2gt_3,478). happensAt(walking(fra2gt_3),478). orientation(fra2gt_4,150,478). coords(fra2gt_4,64,81,478).  visible(fra2gt_4,478). happensAt(walking(fra2gt_4),478).
orientation(fra2gt_3,144,479). coords(fra2gt_3,38,154,479).  visible(fra2gt_3,479). happensAt(walking(fra2gt_3),479). orientation(fra2gt_4,150,479). coords(fra2gt_4,64,81,479).  visible(fra2gt_4,479). happensAt(walking(fra2gt_4),479).
orientation(fra2gt_3,144,480). coords(fra2gt_3,38,153,480).  visible(fra2gt_3,480). happensAt(walking(fra2gt_3),480). orientation(fra2gt_4,150,480). coords(fra2gt_4,64,81,480).  visible(fra2gt_4,480). happensAt(walking(fra2gt_4),480).
orientation(fra2gt_3,144,481). coords(fra2gt_3,38,151,481).  visible(fra2gt_3,481). happensAt(walking(fra2gt_3),481). orientation(fra2gt_4,150,481). coords(fra2gt_4,64,81,481).  visible(fra2gt_4,481). happensAt(walking(fra2gt_4),481).
orientation(fra2gt_3,144,482). coords(fra2gt_3,38,151,482).  visible(fra2gt_3,482). happensAt(walking(fra2gt_3),482). orientation(fra2gt_4,150,482). coords(fra2gt_4,64,81,482).  visible(fra2gt_4,482). happensAt(walking(fra2gt_4),482).
orientation(fra2gt_3,144,483). coords(fra2gt_3,38,149,483).  visible(fra2gt_3,483). happensAt(walking(fra2gt_3),483). orientation(fra2gt_4,150,483). coords(fra2gt_4,64,81,483).  visible(fra2gt_4,483). happensAt(walking(fra2gt_4),483).
orientation(fra2gt_3,144,484). coords(fra2gt_3,38,148,484).  visible(fra2gt_3,484). happensAt(walking(fra2gt_3),484). orientation(fra2gt_4,150,484). coords(fra2gt_4,64,81,484).  visible(fra2gt_4,484). happensAt(walking(fra2gt_4),484).
orientation(fra2gt_3,144,485). coords(fra2gt_3,38,147,485).  visible(fra2gt_3,485). happensAt(walking(fra2gt_3),485). orientation(fra2gt_4,150,485). coords(fra2gt_4,64,81,485).  visible(fra2gt_4,485). happensAt(walking(fra2gt_4),485).
orientation(fra2gt_3,144,486). coords(fra2gt_3,38,147,486).  visible(fra2gt_3,486). happensAt(walking(fra2gt_3),486). orientation(fra2gt_4,150,486). coords(fra2gt_4,64,81,486).  visible(fra2gt_4,486). happensAt(walking(fra2gt_4),486).
orientation(fra2gt_3,144,487). coords(fra2gt_3,38,147,487).  visible(fra2gt_3,487). happensAt(walking(fra2gt_3),487). orientation(fra2gt_4,150,487). coords(fra2gt_4,64,81,487).  visible(fra2gt_4,487). happensAt(walking(fra2gt_4),487).
orientation(fra2gt_3,144,488). coords(fra2gt_3,38,146,488).  visible(fra2gt_3,488). happensAt(walking(fra2gt_3),488). orientation(fra2gt_4,150,488). coords(fra2gt_4,64,81,488).  visible(fra2gt_4,488). happensAt(walking(fra2gt_4),488).
orientation(fra2gt_3,144,489). coords(fra2gt_3,38,146,489).  visible(fra2gt_3,489). happensAt(walking(fra2gt_3),489). orientation(fra2gt_4,150,489). coords(fra2gt_4,64,81,489).  visible(fra2gt_4,489). happensAt(walking(fra2gt_4),489).
orientation(fra2gt_3,144,490). coords(fra2gt_3,39,145,490).  visible(fra2gt_3,490). happensAt(walking(fra2gt_3),490). orientation(fra2gt_4,150,490). coords(fra2gt_4,64,81,490).  visible(fra2gt_4,490). happensAt(walking(fra2gt_4),490).
#show holdsAt/2.