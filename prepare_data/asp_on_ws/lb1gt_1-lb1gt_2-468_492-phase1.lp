
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

	orientation(lb1gt_1,12,468). coords(lb1gt_1,189,233,468).  visible(lb1gt_1,468). happensAt(walking(lb1gt_1),468). orientation(lb1gt_2,146,468). coords(lb1gt_2,188,177,468).  visible(lb1gt_2,468). happensAt(walking(lb1gt_2),468).
orientation(lb1gt_1,12,469). coords(lb1gt_1,191,236,469).  visible(lb1gt_1,469). happensAt(walking(lb1gt_1),469). orientation(lb1gt_2,151,469). coords(lb1gt_2,188,179,469).  visible(lb1gt_2,469). happensAt(walking(lb1gt_2),469).
orientation(lb1gt_1,12,470). coords(lb1gt_1,193,238,470).  visible(lb1gt_1,470). happensAt(walking(lb1gt_1),470). orientation(lb1gt_2,151,470). coords(lb1gt_2,188,180,470).  visible(lb1gt_2,470). happensAt(walking(lb1gt_2),470).
orientation(lb1gt_1,12,471). coords(lb1gt_1,195,239,471).  visible(lb1gt_1,471). happensAt(walking(lb1gt_1),471). orientation(lb1gt_2,151,471). coords(lb1gt_2,188,181,471).  visible(lb1gt_2,471). happensAt(walking(lb1gt_2),471).
orientation(lb1gt_1,12,472). coords(lb1gt_1,198,241,472).  visible(lb1gt_1,472). happensAt(walking(lb1gt_1),472). orientation(lb1gt_2,151,472). coords(lb1gt_2,188,181,472).  visible(lb1gt_2,472). happensAt(walking(lb1gt_2),472).
orientation(lb1gt_1,12,473). coords(lb1gt_1,200,241,473).  visible(lb1gt_1,473). happensAt(walking(lb1gt_1),473). orientation(lb1gt_2,151,473). coords(lb1gt_2,188,182,473).  visible(lb1gt_2,473). happensAt(walking(lb1gt_2),473).
orientation(lb1gt_1,12,474). coords(lb1gt_1,201,244,474).  visible(lb1gt_1,474). happensAt(walking(lb1gt_1),474). orientation(lb1gt_2,151,474). coords(lb1gt_2,188,182,474).  visible(lb1gt_2,474). happensAt(walking(lb1gt_2),474).
orientation(lb1gt_1,12,475). coords(lb1gt_1,202,245,475).  visible(lb1gt_1,475). happensAt(walking(lb1gt_1),475). orientation(lb1gt_2,151,475). coords(lb1gt_2,188,183,475).  visible(lb1gt_2,475). happensAt(walking(lb1gt_2),475).
orientation(lb1gt_1,12,476). coords(lb1gt_1,203,245,476).  visible(lb1gt_1,476). happensAt(walking(lb1gt_1),476). orientation(lb1gt_2,151,476). coords(lb1gt_2,188,184,476).  visible(lb1gt_2,476). happensAt(walking(lb1gt_2),476).
orientation(lb1gt_1,12,477). coords(lb1gt_1,204,246,477).  visible(lb1gt_1,477). happensAt(walking(lb1gt_1),477). orientation(lb1gt_2,160,477). coords(lb1gt_2,189,185,477).  visible(lb1gt_2,477). happensAt(walking(lb1gt_2),477).
orientation(lb1gt_1,12,478). coords(lb1gt_1,205,247,478).  visible(lb1gt_1,478). happensAt(walking(lb1gt_1),478). orientation(lb1gt_2,160,478). coords(lb1gt_2,189,186,478).  visible(lb1gt_2,478). happensAt(walking(lb1gt_2),478).
orientation(lb1gt_1,12,479). coords(lb1gt_1,206,248,479).  visible(lb1gt_1,479). happensAt(walking(lb1gt_1),479). orientation(lb1gt_2,158,479). coords(lb1gt_2,190,189,479).  visible(lb1gt_2,479). happensAt(walking(lb1gt_2),479).
orientation(lb1gt_1,12,480). coords(lb1gt_1,206,252,480).  visible(lb1gt_1,480). happensAt(walking(lb1gt_1),480). orientation(lb1gt_2,159,480). coords(lb1gt_2,191,190,480).  visible(lb1gt_2,480). happensAt(walking(lb1gt_2),480).
orientation(lb1gt_1,12,481). coords(lb1gt_1,207,254,481).  visible(lb1gt_1,481). happensAt(walking(lb1gt_1),481). orientation(lb1gt_2,161,481). coords(lb1gt_2,191,193,481).  visible(lb1gt_2,481). happensAt(walking(lb1gt_2),481).
orientation(lb1gt_1,12,482). coords(lb1gt_1,207,256,482).  visible(lb1gt_1,482). happensAt(walking(lb1gt_1),482). orientation(lb1gt_2,162,482). coords(lb1gt_2,192,196,482).  visible(lb1gt_2,482). happensAt(walking(lb1gt_2),482).
orientation(lb1gt_1,12,483). coords(lb1gt_1,208,259,483).  visible(lb1gt_1,483). happensAt(walking(lb1gt_1),483). orientation(lb1gt_2,171,483). coords(lb1gt_2,193,197,483).  visible(lb1gt_2,483). happensAt(walking(lb1gt_2),483).
orientation(lb1gt_1,12,484). coords(lb1gt_1,208,261,484).  visible(lb1gt_1,484). happensAt(walking(lb1gt_1),484). orientation(lb1gt_2,170,484). coords(lb1gt_2,194,199,484).  visible(lb1gt_2,484). happensAt(walking(lb1gt_2),484).
orientation(lb1gt_1,12,485). coords(lb1gt_1,208,263,485).  visible(lb1gt_1,485). happensAt(walking(lb1gt_1),485). orientation(lb1gt_2,170,485). coords(lb1gt_2,195,201,485).  visible(lb1gt_2,485). happensAt(walking(lb1gt_2),485).
orientation(lb1gt_1,12,486). coords(lb1gt_1,209,264,486).  visible(lb1gt_1,486). happensAt(walking(lb1gt_1),486). orientation(lb1gt_2,172,486). coords(lb1gt_2,195,201,486).  visible(lb1gt_2,486). happensAt(walking(lb1gt_2),486).
orientation(lb1gt_1,12,487). coords(lb1gt_1,209,264,487).  visible(lb1gt_1,487). happensAt(walking(lb1gt_1),487). orientation(lb1gt_2,178,487). coords(lb1gt_2,196,202,487).  visible(lb1gt_2,487). happensAt(walking(lb1gt_2),487).
orientation(lb1gt_1,17,488). coords(lb1gt_1,209,265,488).  visible(lb1gt_1,488). happensAt(walking(lb1gt_1),488). orientation(lb1gt_2,178,488). coords(lb1gt_2,196,202,488).  visible(lb1gt_2,488). happensAt(walking(lb1gt_2),488).
orientation(lb1gt_1,21,489). coords(lb1gt_1,210,267,489).  visible(lb1gt_1,489). happensAt(walking(lb1gt_1),489). orientation(lb1gt_2,0,489). coords(lb1gt_2,197,204,489).  visible(lb1gt_2,489). happensAt(walking(lb1gt_2),489).
orientation(lb1gt_1,23,490). coords(lb1gt_1,211,268,490).  visible(lb1gt_1,490). happensAt(walking(lb1gt_1),490). orientation(lb1gt_2,1,490). coords(lb1gt_2,198,205,490).  visible(lb1gt_2,490). happensAt(walking(lb1gt_2),490).
orientation(lb1gt_1,23,491). coords(lb1gt_1,211,268,491).  visible(lb1gt_1,491). happensAt(walking(lb1gt_1),491). orientation(lb1gt_2,1,491). coords(lb1gt_2,198,207,491).  visible(lb1gt_2,491). happensAt(walking(lb1gt_2),491).
orientation(lb1gt_1,29,492). coords(lb1gt_1,212,268,492).  visible(lb1gt_1,492). happensAt(walking(lb1gt_1),492). orientation(lb1gt_2,1,492). coords(lb1gt_2,199,208,492).  visible(lb1gt_2,492). happensAt(walking(lb1gt_2),492).
#show holdsAt/2.