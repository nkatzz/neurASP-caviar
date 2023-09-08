
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

	orientation(fra1gt_3,175,465). coords(fra1gt_3,103,209,465).  visible(fra1gt_3,465). happensAt(active(fra1gt_3),465). orientation(fra1gt_6,166,465). coords(fra1gt_6,36,194,465).  visible(fra1gt_6,465). happensAt(running(fra1gt_6),465).
orientation(fra1gt_3,175,466). coords(fra1gt_3,103,209,466).  visible(fra1gt_3,466). happensAt(active(fra1gt_3),466). orientation(fra1gt_6,166,466). coords(fra1gt_6,36,195,466).  visible(fra1gt_6,466). happensAt(running(fra1gt_6),466).
orientation(fra1gt_3,175,467). coords(fra1gt_3,103,208,467).  visible(fra1gt_3,467). happensAt(active(fra1gt_3),467). orientation(fra1gt_6,166,467). coords(fra1gt_6,35,196,467).  visible(fra1gt_6,467). happensAt(running(fra1gt_6),467).
orientation(fra1gt_3,175,468). coords(fra1gt_3,103,207,468).  visible(fra1gt_3,468). happensAt(active(fra1gt_3),468). orientation(fra1gt_6,166,468). coords(fra1gt_6,35,196,468).  visible(fra1gt_6,468). happensAt(running(fra1gt_6),468).
orientation(fra1gt_3,175,469). coords(fra1gt_3,103,207,469).  visible(fra1gt_3,469). happensAt(active(fra1gt_3),469). orientation(fra1gt_6,166,469). coords(fra1gt_6,35,197,469).  visible(fra1gt_6,469). happensAt(running(fra1gt_6),469).
orientation(fra1gt_3,175,470). coords(fra1gt_3,103,207,470).  visible(fra1gt_3,470). happensAt(active(fra1gt_3),470). orientation(fra1gt_6,166,470). coords(fra1gt_6,35,199,470).  visible(fra1gt_6,470). happensAt(running(fra1gt_6),470).
orientation(fra1gt_3,175,471). coords(fra1gt_3,103,207,471).  visible(fra1gt_3,471). happensAt(active(fra1gt_3),471). orientation(fra1gt_6,166,471). coords(fra1gt_6,34,199,471).  visible(fra1gt_6,471). happensAt(running(fra1gt_6),471).
orientation(fra1gt_3,175,472). coords(fra1gt_3,103,207,472).  visible(fra1gt_3,472). happensAt(active(fra1gt_3),472). orientation(fra1gt_6,166,472). coords(fra1gt_6,34,200,472).  visible(fra1gt_6,472). happensAt(running(fra1gt_6),472).
orientation(fra1gt_3,175,473). coords(fra1gt_3,103,207,473).  visible(fra1gt_3,473). happensAt(active(fra1gt_3),473). orientation(fra1gt_6,166,473). coords(fra1gt_6,34,202,473).  visible(fra1gt_6,473). happensAt(running(fra1gt_6),473).
orientation(fra1gt_3,175,474). coords(fra1gt_3,103,207,474).  visible(fra1gt_3,474). happensAt(active(fra1gt_3),474). orientation(fra1gt_6,166,474). coords(fra1gt_6,32,203,474).  visible(fra1gt_6,474). happensAt(running(fra1gt_6),474).
orientation(fra1gt_3,175,475). coords(fra1gt_3,103,206,475).  visible(fra1gt_3,475). happensAt(active(fra1gt_3),475). orientation(fra1gt_6,166,475). coords(fra1gt_6,32,204,475).  visible(fra1gt_6,475). happensAt(running(fra1gt_6),475).
orientation(fra1gt_3,175,476). coords(fra1gt_3,102,205,476).  visible(fra1gt_3,476). happensAt(active(fra1gt_3),476). orientation(fra1gt_6,166,476). coords(fra1gt_6,32,204,476).  visible(fra1gt_6,476). happensAt(running(fra1gt_6),476).
orientation(fra1gt_3,175,477). coords(fra1gt_3,102,205,477).  visible(fra1gt_3,477). happensAt(active(fra1gt_3),477). orientation(fra1gt_6,166,477). coords(fra1gt_6,32,204,477).  visible(fra1gt_6,477). happensAt(running(fra1gt_6),477).
orientation(fra1gt_3,175,478). coords(fra1gt_3,104,205,478).  visible(fra1gt_3,478). happensAt(active(fra1gt_3),478). orientation(fra1gt_6,166,478). coords(fra1gt_6,32,206,478).  visible(fra1gt_6,478). happensAt(running(fra1gt_6),478).
orientation(fra1gt_3,175,479). coords(fra1gt_3,104,205,479).  visible(fra1gt_3,479). happensAt(active(fra1gt_3),479). orientation(fra1gt_6,166,479). coords(fra1gt_6,31,206,479).  visible(fra1gt_6,479). happensAt(running(fra1gt_6),479).
orientation(fra1gt_3,175,480). coords(fra1gt_3,104,205,480).  visible(fra1gt_3,480). happensAt(active(fra1gt_3),480). orientation(fra1gt_6,166,480). coords(fra1gt_6,31,206,480).  visible(fra1gt_6,480). happensAt(running(fra1gt_6),480).
orientation(fra1gt_3,175,481). coords(fra1gt_3,104,204,481).  visible(fra1gt_3,481). happensAt(active(fra1gt_3),481). orientation(fra1gt_6,166,481). coords(fra1gt_6,30,207,481).  visible(fra1gt_6,481). happensAt(running(fra1gt_6),481).
orientation(fra1gt_3,175,482). coords(fra1gt_3,104,204,482).  visible(fra1gt_3,482). happensAt(active(fra1gt_3),482). orientation(fra1gt_6,166,482). coords(fra1gt_6,30,208,482).  visible(fra1gt_6,482). happensAt(running(fra1gt_6),482).
orientation(fra1gt_3,175,483). coords(fra1gt_3,104,204,483).  visible(fra1gt_3,483). happensAt(active(fra1gt_3),483). orientation(fra1gt_6,166,483). coords(fra1gt_6,30,208,483).  visible(fra1gt_6,483). happensAt(running(fra1gt_6),483).
orientation(fra1gt_3,175,484). coords(fra1gt_3,104,204,484).  visible(fra1gt_3,484). happensAt(active(fra1gt_3),484). orientation(fra1gt_6,166,484). coords(fra1gt_6,30,208,484).  visible(fra1gt_6,484). happensAt(running(fra1gt_6),484).
orientation(fra1gt_3,175,485). coords(fra1gt_3,104,204,485).  visible(fra1gt_3,485). happensAt(active(fra1gt_3),485). orientation(fra1gt_6,166,485). coords(fra1gt_6,30,208,485).  visible(fra1gt_6,485). happensAt(running(fra1gt_6),485).
orientation(fra1gt_3,175,486). coords(fra1gt_3,104,204,486).  visible(fra1gt_3,486). happensAt(active(fra1gt_3),486). orientation(fra1gt_6,166,486). coords(fra1gt_6,26,209,486).  visible(fra1gt_6,486). happensAt(running(fra1gt_6),486).
orientation(fra1gt_3,175,487). coords(fra1gt_3,104,204,487).  visible(fra1gt_3,487). happensAt(active(fra1gt_3),487). orientation(fra1gt_6,166,487). coords(fra1gt_6,26,210,487).  visible(fra1gt_6,487). happensAt(running(fra1gt_6),487).
orientation(fra1gt_3,175,488). coords(fra1gt_3,101,204,488).  visible(fra1gt_3,488). happensAt(active(fra1gt_3),488). orientation(fra1gt_6,166,488). coords(fra1gt_6,25,211,488).  visible(fra1gt_6,488). happensAt(running(fra1gt_6),488).
orientation(fra1gt_3,175,489). coords(fra1gt_3,101,204,489).  visible(fra1gt_3,489). happensAt(active(fra1gt_3),489). orientation(fra1gt_6,166,489). coords(fra1gt_6,24,211,489).  happensAt(disappear(fra1gt_6),489). happensAt(running(fra1gt_6),489).
#show holdsAt/2.