
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

	orientation(mwt2gt_1,164,469). coords(mwt2gt_1,66,128,469).  visible(mwt2gt_1,469). happensAt(walking(mwt2gt_1),469). orientation(mwt2gt_2,157,469). coords(mwt2gt_2,65,146,469).  visible(mwt2gt_2,469). happensAt(walking(mwt2gt_2),469).
orientation(mwt2gt_1,164,470). coords(mwt2gt_1,66,128,470).  visible(mwt2gt_1,470). happensAt(walking(mwt2gt_1),470). orientation(mwt2gt_2,157,470). coords(mwt2gt_2,65,146,470).  visible(mwt2gt_2,470). happensAt(walking(mwt2gt_2),470).
orientation(mwt2gt_1,164,471). coords(mwt2gt_1,65,129,471).  visible(mwt2gt_1,471). happensAt(walking(mwt2gt_1),471). orientation(mwt2gt_2,157,471). coords(mwt2gt_2,65,147,471).  visible(mwt2gt_2,471). happensAt(walking(mwt2gt_2),471).
orientation(mwt2gt_1,164,472). coords(mwt2gt_1,65,129,472).  visible(mwt2gt_1,472). happensAt(walking(mwt2gt_1),472). orientation(mwt2gt_2,144,472). coords(mwt2gt_2,64,148,472).  visible(mwt2gt_2,472). happensAt(walking(mwt2gt_2),472).
orientation(mwt2gt_1,164,473). coords(mwt2gt_1,64,129,473).  visible(mwt2gt_1,473). happensAt(walking(mwt2gt_1),473). orientation(mwt2gt_2,144,473). coords(mwt2gt_2,64,148,473).  visible(mwt2gt_2,473). happensAt(walking(mwt2gt_2),473).
orientation(mwt2gt_1,164,474). coords(mwt2gt_1,64,130,474).  visible(mwt2gt_1,474). happensAt(walking(mwt2gt_1),474). orientation(mwt2gt_2,152,474). coords(mwt2gt_2,63,149,474).  visible(mwt2gt_2,474). happensAt(walking(mwt2gt_2),474).
orientation(mwt2gt_1,164,475). coords(mwt2gt_1,64,132,475).  visible(mwt2gt_1,475). happensAt(walking(mwt2gt_1),475). orientation(mwt2gt_2,153,475). coords(mwt2gt_2,62,149,475).  visible(mwt2gt_2,475). happensAt(walking(mwt2gt_2),475).
orientation(mwt2gt_1,164,476). coords(mwt2gt_1,64,134,476).  visible(mwt2gt_1,476). happensAt(walking(mwt2gt_1),476). orientation(mwt2gt_2,153,476). coords(mwt2gt_2,61,150,476).  visible(mwt2gt_2,476). happensAt(walking(mwt2gt_2),476).
orientation(mwt2gt_1,164,477). coords(mwt2gt_1,64,135,477).  visible(mwt2gt_1,477). happensAt(walking(mwt2gt_1),477). orientation(mwt2gt_2,150,477). coords(mwt2gt_2,60,150,477).  visible(mwt2gt_2,477). happensAt(walking(mwt2gt_2),477).
orientation(mwt2gt_1,164,478). coords(mwt2gt_1,63,137,478).  visible(mwt2gt_1,478). happensAt(walking(mwt2gt_1),478). orientation(mwt2gt_2,157,478). coords(mwt2gt_2,59,150,478).  visible(mwt2gt_2,478). happensAt(walking(mwt2gt_2),478).
orientation(mwt2gt_1,164,479). coords(mwt2gt_1,64,137,479).  visible(mwt2gt_1,479). happensAt(walking(mwt2gt_1),479). orientation(mwt2gt_2,162,479). coords(mwt2gt_2,59,150,479).  visible(mwt2gt_2,479). happensAt(walking(mwt2gt_2),479).
orientation(mwt2gt_1,164,480). coords(mwt2gt_1,63,138,480).  visible(mwt2gt_1,480). happensAt(walking(mwt2gt_1),480). orientation(mwt2gt_2,161,480). coords(mwt2gt_2,59,150,480).  visible(mwt2gt_2,480). happensAt(walking(mwt2gt_2),480).
orientation(mwt2gt_1,164,481). coords(mwt2gt_1,62,138,481).  visible(mwt2gt_1,481). happensAt(walking(mwt2gt_1),481). orientation(mwt2gt_2,160,481). coords(mwt2gt_2,59,151,481).  visible(mwt2gt_2,481). happensAt(walking(mwt2gt_2),481).
orientation(mwt2gt_1,164,482). coords(mwt2gt_1,61,138,482).  visible(mwt2gt_1,482). happensAt(walking(mwt2gt_1),482). orientation(mwt2gt_2,160,482). coords(mwt2gt_2,59,151,482).  visible(mwt2gt_2,482). happensAt(walking(mwt2gt_2),482).
orientation(mwt2gt_1,164,483). coords(mwt2gt_1,61,139,483).  visible(mwt2gt_1,483). happensAt(walking(mwt2gt_1),483). orientation(mwt2gt_2,147,483). coords(mwt2gt_2,59,153,483).  visible(mwt2gt_2,483). happensAt(walking(mwt2gt_2),483).
orientation(mwt2gt_1,156,484). coords(mwt2gt_1,61,139,484).  visible(mwt2gt_1,484). happensAt(walking(mwt2gt_1),484). orientation(mwt2gt_2,140,484). coords(mwt2gt_2,58,154,484).  visible(mwt2gt_2,484). happensAt(walking(mwt2gt_2),484).
orientation(mwt2gt_1,156,485). coords(mwt2gt_1,60,140,485).  visible(mwt2gt_1,485). happensAt(walking(mwt2gt_1),485). orientation(mwt2gt_2,142,485). coords(mwt2gt_2,58,154,485).  visible(mwt2gt_2,485). happensAt(walking(mwt2gt_2),485).
orientation(mwt2gt_1,156,486). coords(mwt2gt_1,60,142,486).  visible(mwt2gt_1,486). happensAt(walking(mwt2gt_1),486). orientation(mwt2gt_2,147,486). coords(mwt2gt_2,58,154,486).  visible(mwt2gt_2,486). happensAt(walking(mwt2gt_2),486).
orientation(mwt2gt_1,156,487). coords(mwt2gt_1,60,144,487).  visible(mwt2gt_1,487). happensAt(walking(mwt2gt_1),487). orientation(mwt2gt_2,152,487). coords(mwt2gt_2,58,155,487).  visible(mwt2gt_2,487). happensAt(walking(mwt2gt_2),487).
orientation(mwt2gt_1,156,488). coords(mwt2gt_1,60,145,488).  visible(mwt2gt_1,488). happensAt(walking(mwt2gt_1),488). orientation(mwt2gt_2,149,488). coords(mwt2gt_2,57,155,488).  visible(mwt2gt_2,488). happensAt(walking(mwt2gt_2),488).
orientation(mwt2gt_1,156,489). coords(mwt2gt_1,60,146,489).  visible(mwt2gt_1,489). happensAt(walking(mwt2gt_1),489). orientation(mwt2gt_2,146,489). coords(mwt2gt_2,56,155,489).  visible(mwt2gt_2,489). happensAt(walking(mwt2gt_2),489).
orientation(mwt2gt_1,156,490). coords(mwt2gt_1,59,146,490).  visible(mwt2gt_1,490). happensAt(walking(mwt2gt_1),490). orientation(mwt2gt_2,151,490). coords(mwt2gt_2,56,155,490).  visible(mwt2gt_2,490). happensAt(walking(mwt2gt_2),490).
orientation(mwt2gt_1,156,491). coords(mwt2gt_1,59,147,491).  visible(mwt2gt_1,491). happensAt(walking(mwt2gt_1),491). orientation(mwt2gt_2,147,491). coords(mwt2gt_2,56,155,491).  visible(mwt2gt_2,491). happensAt(walking(mwt2gt_2),491).
orientation(mwt2gt_1,156,492). coords(mwt2gt_1,59,148,492).  visible(mwt2gt_1,492). happensAt(walking(mwt2gt_1),492). orientation(mwt2gt_2,145,492). coords(mwt2gt_2,56,157,492).  visible(mwt2gt_2,492). happensAt(walking(mwt2gt_2),492).
orientation(mwt2gt_1,156,493). coords(mwt2gt_1,59,148,493).  visible(mwt2gt_1,493). happensAt(walking(mwt2gt_1),493). orientation(mwt2gt_2,146,493). coords(mwt2gt_2,53,158,493).  visible(mwt2gt_2,493). happensAt(walking(mwt2gt_2),493).
#show holdsAt/2.