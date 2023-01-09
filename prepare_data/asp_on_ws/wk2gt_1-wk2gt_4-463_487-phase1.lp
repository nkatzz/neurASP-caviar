
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

	orientation(wk2gt_1,125,463). coords(wk2gt_1,115,53,463).  visible(wk2gt_1,463). happensAt(inactive(wk2gt_1),463). orientation(wk2gt_4,149,463). coords(wk2gt_4,84,98,463).  visible(wk2gt_4,463). happensAt(walking(wk2gt_4),463).
orientation(wk2gt_1,125,464). coords(wk2gt_1,115,53,464).  visible(wk2gt_1,464). happensAt(inactive(wk2gt_1),464). orientation(wk2gt_4,149,464). coords(wk2gt_4,83,98,464).  visible(wk2gt_4,464). happensAt(walking(wk2gt_4),464).
orientation(wk2gt_1,125,465). coords(wk2gt_1,115,53,465).  visible(wk2gt_1,465). happensAt(inactive(wk2gt_1),465). orientation(wk2gt_4,149,465). coords(wk2gt_4,82,98,465).  visible(wk2gt_4,465). happensAt(walking(wk2gt_4),465).
orientation(wk2gt_1,125,466). coords(wk2gt_1,115,53,466).  visible(wk2gt_1,466). happensAt(inactive(wk2gt_1),466). orientation(wk2gt_4,149,466). coords(wk2gt_4,81,98,466).  visible(wk2gt_4,466). happensAt(walking(wk2gt_4),466).
orientation(wk2gt_1,125,467). coords(wk2gt_1,115,53,467).  visible(wk2gt_1,467). happensAt(inactive(wk2gt_1),467). orientation(wk2gt_4,149,467). coords(wk2gt_4,80,98,467).  visible(wk2gt_4,467). happensAt(walking(wk2gt_4),467).
orientation(wk2gt_1,125,468). coords(wk2gt_1,115,53,468).  visible(wk2gt_1,468). happensAt(inactive(wk2gt_1),468). orientation(wk2gt_4,149,468). coords(wk2gt_4,80,98,468).  visible(wk2gt_4,468). happensAt(walking(wk2gt_4),468).
orientation(wk2gt_1,125,469). coords(wk2gt_1,115,53,469).  visible(wk2gt_1,469). happensAt(inactive(wk2gt_1),469). orientation(wk2gt_4,149,469). coords(wk2gt_4,80,98,469).  visible(wk2gt_4,469). happensAt(walking(wk2gt_4),469).
orientation(wk2gt_1,125,470). coords(wk2gt_1,115,53,470).  visible(wk2gt_1,470). happensAt(inactive(wk2gt_1),470). orientation(wk2gt_4,149,470). coords(wk2gt_4,80,98,470).  visible(wk2gt_4,470). happensAt(walking(wk2gt_4),470).
orientation(wk2gt_1,125,471). coords(wk2gt_1,115,53,471).  visible(wk2gt_1,471). happensAt(inactive(wk2gt_1),471). orientation(wk2gt_4,149,471). coords(wk2gt_4,80,98,471).  visible(wk2gt_4,471). happensAt(walking(wk2gt_4),471).
orientation(wk2gt_1,125,472). coords(wk2gt_1,115,53,472).  visible(wk2gt_1,472). happensAt(inactive(wk2gt_1),472). orientation(wk2gt_4,149,472). coords(wk2gt_4,80,98,472).  visible(wk2gt_4,472). happensAt(walking(wk2gt_4),472).
orientation(wk2gt_1,125,473). coords(wk2gt_1,115,53,473).  visible(wk2gt_1,473). happensAt(inactive(wk2gt_1),473). orientation(wk2gt_4,149,473). coords(wk2gt_4,79,98,473).  visible(wk2gt_4,473). happensAt(walking(wk2gt_4),473).
orientation(wk2gt_1,125,474). coords(wk2gt_1,115,53,474).  visible(wk2gt_1,474). happensAt(inactive(wk2gt_1),474). orientation(wk2gt_4,149,474). coords(wk2gt_4,78,98,474).  visible(wk2gt_4,474). happensAt(walking(wk2gt_4),474).
orientation(wk2gt_1,125,475). coords(wk2gt_1,115,53,475).  visible(wk2gt_1,475). happensAt(inactive(wk2gt_1),475). orientation(wk2gt_4,149,475). coords(wk2gt_4,77,98,475).  visible(wk2gt_4,475). happensAt(walking(wk2gt_4),475).
orientation(wk2gt_1,125,476). coords(wk2gt_1,115,53,476).  visible(wk2gt_1,476). happensAt(inactive(wk2gt_1),476). orientation(wk2gt_4,149,476). coords(wk2gt_4,76,97,476).  visible(wk2gt_4,476). happensAt(walking(wk2gt_4),476).
orientation(wk2gt_1,125,477). coords(wk2gt_1,115,53,477).  visible(wk2gt_1,477). happensAt(inactive(wk2gt_1),477). orientation(wk2gt_4,149,477). coords(wk2gt_4,75,97,477).  visible(wk2gt_4,477). happensAt(walking(wk2gt_4),477).
orientation(wk2gt_1,125,478). coords(wk2gt_1,115,53,478).  visible(wk2gt_1,478). happensAt(inactive(wk2gt_1),478). orientation(wk2gt_4,146,478). coords(wk2gt_4,75,97,478).  visible(wk2gt_4,478). happensAt(walking(wk2gt_4),478).
orientation(wk2gt_1,125,479). coords(wk2gt_1,115,53,479).  visible(wk2gt_1,479). happensAt(inactive(wk2gt_1),479). orientation(wk2gt_4,146,479). coords(wk2gt_4,75,97,479).  visible(wk2gt_4,479). happensAt(walking(wk2gt_4),479).
orientation(wk2gt_1,125,480). coords(wk2gt_1,115,53,480).  visible(wk2gt_1,480). happensAt(inactive(wk2gt_1),480). orientation(wk2gt_4,146,480). coords(wk2gt_4,75,97,480).  visible(wk2gt_4,480). happensAt(walking(wk2gt_4),480).
orientation(wk2gt_1,125,481). coords(wk2gt_1,115,53,481).  visible(wk2gt_1,481). happensAt(inactive(wk2gt_1),481). orientation(wk2gt_4,146,481). coords(wk2gt_4,74,97,481).  visible(wk2gt_4,481). happensAt(walking(wk2gt_4),481).
orientation(wk2gt_1,125,482). coords(wk2gt_1,115,53,482).  visible(wk2gt_1,482). happensAt(inactive(wk2gt_1),482). orientation(wk2gt_4,146,482). coords(wk2gt_4,74,97,482).  visible(wk2gt_4,482). happensAt(walking(wk2gt_4),482).
orientation(wk2gt_1,125,483). coords(wk2gt_1,115,53,483).  visible(wk2gt_1,483). happensAt(inactive(wk2gt_1),483). orientation(wk2gt_4,146,483). coords(wk2gt_4,74,97,483).  visible(wk2gt_4,483). happensAt(walking(wk2gt_4),483).
orientation(wk2gt_1,125,484). coords(wk2gt_1,115,53,484).  visible(wk2gt_1,484). happensAt(inactive(wk2gt_1),484). orientation(wk2gt_4,146,484). coords(wk2gt_4,74,97,484).  visible(wk2gt_4,484). happensAt(walking(wk2gt_4),484).
orientation(wk2gt_1,125,485). coords(wk2gt_1,115,53,485).  visible(wk2gt_1,485). happensAt(inactive(wk2gt_1),485). orientation(wk2gt_4,146,485). coords(wk2gt_4,73,97,485).  visible(wk2gt_4,485). happensAt(walking(wk2gt_4),485).
orientation(wk2gt_1,125,486). coords(wk2gt_1,115,53,486).  visible(wk2gt_1,486). happensAt(inactive(wk2gt_1),486). orientation(wk2gt_4,146,486). coords(wk2gt_4,73,97,486).  visible(wk2gt_4,486). happensAt(walking(wk2gt_4),486).
orientation(wk2gt_1,125,487). coords(wk2gt_1,115,53,487).  visible(wk2gt_1,487). happensAt(inactive(wk2gt_1),487). orientation(wk2gt_4,146,487). coords(wk2gt_4,72,97,487).  visible(wk2gt_4,487). happensAt(walking(wk2gt_4),487).
#show holdsAt/2.