
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

	orientation(lb2gt_4,154,467). coords(lb2gt_4,65,123,467).  visible(lb2gt_4,467). happensAt(walking(lb2gt_4),467). orientation(lb2gt_6,151,467). coords(lb2gt_6,72,122,467).  visible(lb2gt_6,467). happensAt(inactive(lb2gt_6),467).
orientation(lb2gt_4,154,468). coords(lb2gt_4,65,123,468).  visible(lb2gt_4,468). happensAt(walking(lb2gt_4),468). orientation(lb2gt_6,151,468). coords(lb2gt_6,72,122,468).  visible(lb2gt_6,468). happensAt(inactive(lb2gt_6),468).
orientation(lb2gt_4,154,469). coords(lb2gt_4,66,122,469).  visible(lb2gt_4,469). happensAt(walking(lb2gt_4),469). orientation(lb2gt_6,151,469). coords(lb2gt_6,72,122,469).  visible(lb2gt_6,469). happensAt(inactive(lb2gt_6),469).
orientation(lb2gt_4,154,470). coords(lb2gt_4,66,122,470).  visible(lb2gt_4,470). happensAt(walking(lb2gt_4),470). orientation(lb2gt_6,151,470). coords(lb2gt_6,72,122,470).  visible(lb2gt_6,470). happensAt(inactive(lb2gt_6),470).
orientation(lb2gt_4,154,471). coords(lb2gt_4,66,122,471).  visible(lb2gt_4,471). happensAt(walking(lb2gt_4),471). orientation(lb2gt_6,151,471). coords(lb2gt_6,72,122,471).  visible(lb2gt_6,471). happensAt(inactive(lb2gt_6),471).
orientation(lb2gt_4,154,472). coords(lb2gt_4,66,122,472).  visible(lb2gt_4,472). happensAt(walking(lb2gt_4),472). orientation(lb2gt_6,151,472). coords(lb2gt_6,72,122,472).  visible(lb2gt_6,472). happensAt(inactive(lb2gt_6),472).
orientation(lb2gt_4,154,473). coords(lb2gt_4,67,121,473).  visible(lb2gt_4,473). happensAt(walking(lb2gt_4),473). orientation(lb2gt_6,151,473). coords(lb2gt_6,72,122,473).  visible(lb2gt_6,473). happensAt(inactive(lb2gt_6),473).
orientation(lb2gt_4,154,474). coords(lb2gt_4,67,121,474).  visible(lb2gt_4,474). happensAt(walking(lb2gt_4),474). orientation(lb2gt_6,151,474). coords(lb2gt_6,72,122,474).  visible(lb2gt_6,474). happensAt(inactive(lb2gt_6),474).
orientation(lb2gt_4,154,475). coords(lb2gt_4,67,121,475).  visible(lb2gt_4,475). happensAt(walking(lb2gt_4),475). orientation(lb2gt_6,151,475). coords(lb2gt_6,72,122,475).  visible(lb2gt_6,475). happensAt(inactive(lb2gt_6),475).
orientation(lb2gt_4,154,476). coords(lb2gt_4,67,121,476).  visible(lb2gt_4,476). happensAt(walking(lb2gt_4),476). orientation(lb2gt_6,151,476). coords(lb2gt_6,72,122,476).  visible(lb2gt_6,476). happensAt(inactive(lb2gt_6),476).
orientation(lb2gt_4,154,477). coords(lb2gt_4,67,120,477).  visible(lb2gt_4,477). happensAt(walking(lb2gt_4),477). orientation(lb2gt_6,151,477). coords(lb2gt_6,72,122,477).  visible(lb2gt_6,477). happensAt(inactive(lb2gt_6),477).
orientation(lb2gt_4,154,478). coords(lb2gt_4,68,119,478).  visible(lb2gt_4,478). happensAt(walking(lb2gt_4),478). orientation(lb2gt_6,151,478). coords(lb2gt_6,72,122,478).  visible(lb2gt_6,478). happensAt(inactive(lb2gt_6),478).
orientation(lb2gt_4,154,479). coords(lb2gt_4,68,118,479).  visible(lb2gt_4,479). happensAt(walking(lb2gt_4),479). orientation(lb2gt_6,151,479). coords(lb2gt_6,72,122,479).  visible(lb2gt_6,479). happensAt(inactive(lb2gt_6),479).
orientation(lb2gt_4,154,480). coords(lb2gt_4,68,118,480).  visible(lb2gt_4,480). happensAt(walking(lb2gt_4),480). orientation(lb2gt_6,151,480). coords(lb2gt_6,72,122,480).  visible(lb2gt_6,480). happensAt(inactive(lb2gt_6),480).
orientation(lb2gt_4,154,481). coords(lb2gt_4,68,116,481).  visible(lb2gt_4,481). happensAt(walking(lb2gt_4),481). orientation(lb2gt_6,151,481). coords(lb2gt_6,72,122,481).  visible(lb2gt_6,481). happensAt(inactive(lb2gt_6),481).
orientation(lb2gt_4,154,482). coords(lb2gt_4,68,115,482).  visible(lb2gt_4,482). happensAt(walking(lb2gt_4),482). orientation(lb2gt_6,151,482). coords(lb2gt_6,72,122,482).  visible(lb2gt_6,482). happensAt(inactive(lb2gt_6),482).
orientation(lb2gt_4,154,483). coords(lb2gt_4,68,115,483).  visible(lb2gt_4,483). happensAt(walking(lb2gt_4),483). orientation(lb2gt_6,151,483). coords(lb2gt_6,72,122,483).  visible(lb2gt_6,483). happensAt(inactive(lb2gt_6),483).
orientation(lb2gt_4,154,484). coords(lb2gt_4,68,115,484).  visible(lb2gt_4,484). happensAt(walking(lb2gt_4),484). orientation(lb2gt_6,151,484). coords(lb2gt_6,72,122,484).  visible(lb2gt_6,484). happensAt(inactive(lb2gt_6),484).
orientation(lb2gt_4,154,485). coords(lb2gt_4,68,114,485).  visible(lb2gt_4,485). happensAt(walking(lb2gt_4),485). orientation(lb2gt_6,151,485). coords(lb2gt_6,72,122,485).  visible(lb2gt_6,485). happensAt(inactive(lb2gt_6),485).
orientation(lb2gt_4,154,486). coords(lb2gt_4,69,114,486).  visible(lb2gt_4,486). happensAt(walking(lb2gt_4),486). orientation(lb2gt_6,151,486). coords(lb2gt_6,72,122,486).  visible(lb2gt_6,486). happensAt(inactive(lb2gt_6),486).
orientation(lb2gt_4,154,487). coords(lb2gt_4,69,113,487).  visible(lb2gt_4,487). happensAt(walking(lb2gt_4),487). orientation(lb2gt_6,151,487). coords(lb2gt_6,72,122,487).  visible(lb2gt_6,487). happensAt(inactive(lb2gt_6),487).
orientation(lb2gt_4,154,488). coords(lb2gt_4,70,113,488).  visible(lb2gt_4,488). happensAt(walking(lb2gt_4),488). orientation(lb2gt_6,151,488). coords(lb2gt_6,72,122,488).  visible(lb2gt_6,488). happensAt(inactive(lb2gt_6),488).
orientation(lb2gt_4,154,489). coords(lb2gt_4,71,113,489).  visible(lb2gt_4,489). happensAt(walking(lb2gt_4),489). orientation(lb2gt_6,151,489). coords(lb2gt_6,72,122,489).  visible(lb2gt_6,489). happensAt(inactive(lb2gt_6),489).
orientation(lb2gt_4,154,490). coords(lb2gt_4,71,112,490).  visible(lb2gt_4,490). happensAt(walking(lb2gt_4),490). orientation(lb2gt_6,151,490). coords(lb2gt_6,72,122,490).  visible(lb2gt_6,490). happensAt(inactive(lb2gt_6),490).
orientation(lb2gt_4,154,491). coords(lb2gt_4,72,111,491).  visible(lb2gt_4,491). happensAt(walking(lb2gt_4),491). orientation(lb2gt_6,151,491). coords(lb2gt_6,72,122,491).  visible(lb2gt_6,491). happensAt(inactive(lb2gt_6),491).
#show holdsAt/2.