
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
orientation(bww1gt_1,0,483). coords(bww1gt_1,105,264,483).  visible(bww1gt_1,483). happensAt(inactive(bww1gt_1),483). orientation(bww1gt_2,160,483). coords(bww1gt_2,174,168,483).  visible(bww1gt_2,483). happensAt(walking(bww1gt_2),483).
orientation(bww1gt_1,0,484). coords(bww1gt_1,105,264,484).  visible(bww1gt_1,484). happensAt(inactive(bww1gt_1),484). orientation(bww1gt_2,158,484). coords(bww1gt_2,175,168,484).  visible(bww1gt_2,484). happensAt(walking(bww1gt_2),484).
orientation(bww1gt_1,0,485). coords(bww1gt_1,105,264,485).  visible(bww1gt_1,485). happensAt(inactive(bww1gt_1),485). orientation(bww1gt_2,158,485). coords(bww1gt_2,176,170,485).  visible(bww1gt_2,485). happensAt(walking(bww1gt_2),485).
orientation(bww1gt_1,0,486). coords(bww1gt_1,105,264,486).  visible(bww1gt_1,486). happensAt(inactive(bww1gt_1),486). orientation(bww1gt_2,154,486). coords(bww1gt_2,177,172,486).  visible(bww1gt_2,486). happensAt(walking(bww1gt_2),486).
orientation(bww1gt_1,0,487). coords(bww1gt_1,105,264,487).  visible(bww1gt_1,487). happensAt(inactive(bww1gt_1),487). orientation(bww1gt_2,153,487). coords(bww1gt_2,178,174,487).  visible(bww1gt_2,487). happensAt(walking(bww1gt_2),487).
orientation(bww1gt_1,0,488). coords(bww1gt_1,105,264,488).  visible(bww1gt_1,488). happensAt(inactive(bww1gt_1),488). orientation(bww1gt_2,155,488). coords(bww1gt_2,178,175,488).  visible(bww1gt_2,488). happensAt(walking(bww1gt_2),488).
orientation(bww1gt_1,0,489). coords(bww1gt_1,105,264,489).  visible(bww1gt_1,489). happensAt(inactive(bww1gt_1),489). orientation(bww1gt_2,157,489). coords(bww1gt_2,179,176,489).  visible(bww1gt_2,489). happensAt(walking(bww1gt_2),489).
orientation(bww1gt_1,0,490). coords(bww1gt_1,105,264,490).  visible(bww1gt_1,490). happensAt(inactive(bww1gt_1),490). orientation(bww1gt_2,155,490). coords(bww1gt_2,179,177,490).  visible(bww1gt_2,490). happensAt(walking(bww1gt_2),490).
orientation(bww1gt_1,0,491). coords(bww1gt_1,105,264,491).  visible(bww1gt_1,491). happensAt(inactive(bww1gt_1),491). orientation(bww1gt_2,152,491). coords(bww1gt_2,179,177,491).  visible(bww1gt_2,491). happensAt(walking(bww1gt_2),491).
orientation(bww1gt_1,0,492). coords(bww1gt_1,105,264,492).  visible(bww1gt_1,492). happensAt(inactive(bww1gt_1),492). orientation(bww1gt_2,157,492). coords(bww1gt_2,179,177,492).  visible(bww1gt_2,492). happensAt(walking(bww1gt_2),492).
orientation(bww1gt_1,0,493). coords(bww1gt_1,105,264,493).  visible(bww1gt_1,493). happensAt(inactive(bww1gt_1),493). orientation(bww1gt_2,157,493). coords(bww1gt_2,179,178,493).  visible(bww1gt_2,493). happensAt(walking(bww1gt_2),493).
orientation(bww1gt_1,0,494). coords(bww1gt_1,105,264,494).  visible(bww1gt_1,494). happensAt(inactive(bww1gt_1),494). orientation(bww1gt_2,154,494). coords(bww1gt_2,179,178,494).  visible(bww1gt_2,494). happensAt(walking(bww1gt_2),494).
#show holdsAt/2.