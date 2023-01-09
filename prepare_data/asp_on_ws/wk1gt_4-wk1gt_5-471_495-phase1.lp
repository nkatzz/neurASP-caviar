
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
orientation(wk1gt_4,147,484). coords(wk1gt_4,56,168,484).  visible(wk1gt_4,484). happensAt(active(wk1gt_4),484). orientation(wk1gt_5,162,484). coords(wk1gt_5,57,152,484).  visible(wk1gt_5,484). happensAt(active(wk1gt_5),484).
orientation(wk1gt_4,147,485). coords(wk1gt_4,56,169,485).  visible(wk1gt_4,485). happensAt(active(wk1gt_4),485). orientation(wk1gt_5,162,485). coords(wk1gt_5,57,152,485).  visible(wk1gt_5,485). happensAt(active(wk1gt_5),485).
orientation(wk1gt_4,147,486). coords(wk1gt_4,56,168,486).  visible(wk1gt_4,486). happensAt(active(wk1gt_4),486). orientation(wk1gt_5,162,486). coords(wk1gt_5,57,152,486).  visible(wk1gt_5,486). happensAt(active(wk1gt_5),486).
orientation(wk1gt_4,153,487). coords(wk1gt_4,56,168,487).  visible(wk1gt_4,487). happensAt(active(wk1gt_4),487). orientation(wk1gt_5,162,487). coords(wk1gt_5,57,152,487).  visible(wk1gt_5,487). happensAt(active(wk1gt_5),487).
orientation(wk1gt_4,154,488). coords(wk1gt_4,56,168,488).  visible(wk1gt_4,488). happensAt(active(wk1gt_4),488). orientation(wk1gt_5,162,488). coords(wk1gt_5,57,152,488).  visible(wk1gt_5,488). happensAt(active(wk1gt_5),488).
orientation(wk1gt_4,154,489). coords(wk1gt_4,56,168,489).  visible(wk1gt_4,489). happensAt(active(wk1gt_4),489). orientation(wk1gt_5,162,489). coords(wk1gt_5,57,151,489).  visible(wk1gt_5,489). happensAt(active(wk1gt_5),489).
orientation(wk1gt_4,154,490). coords(wk1gt_4,56,168,490).  visible(wk1gt_4,490). happensAt(active(wk1gt_4),490). orientation(wk1gt_5,162,490). coords(wk1gt_5,57,150,490).  visible(wk1gt_5,490). happensAt(active(wk1gt_5),490).
orientation(wk1gt_4,157,491). coords(wk1gt_4,56,169,491).  visible(wk1gt_4,491). happensAt(active(wk1gt_4),491). orientation(wk1gt_5,162,491). coords(wk1gt_5,57,150,491).  visible(wk1gt_5,491). happensAt(active(wk1gt_5),491).
orientation(wk1gt_4,157,492). coords(wk1gt_4,56,169,492).  visible(wk1gt_4,492). happensAt(active(wk1gt_4),492). orientation(wk1gt_5,162,492). coords(wk1gt_5,57,150,492).  visible(wk1gt_5,492). happensAt(active(wk1gt_5),492).
orientation(wk1gt_4,157,493). coords(wk1gt_4,56,169,493).  visible(wk1gt_4,493). happensAt(active(wk1gt_4),493). orientation(wk1gt_5,162,493). coords(wk1gt_5,57,150,493).  visible(wk1gt_5,493). happensAt(active(wk1gt_5),493).
orientation(wk1gt_4,157,494). coords(wk1gt_4,56,170,494).  visible(wk1gt_4,494). happensAt(active(wk1gt_4),494). orientation(wk1gt_5,162,494). coords(wk1gt_5,57,150,494).  visible(wk1gt_5,494). happensAt(active(wk1gt_5),494).
orientation(wk1gt_4,157,495). coords(wk1gt_4,56,170,495).  visible(wk1gt_4,495). happensAt(active(wk1gt_4),495). orientation(wk1gt_5,162,495). coords(wk1gt_5,57,150,495).  visible(wk1gt_5,495). happensAt(active(wk1gt_5),495).
#show holdsAt/2.