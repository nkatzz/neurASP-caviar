
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

	orientation(lb1gt_1,176,456). coords(lb1gt_1,181,217,456).  visible(lb1gt_1,456). happensAt(walking(lb1gt_1),456). orientation(lb1gt_2,146,456). coords(lb1gt_2,187,160,456).  visible(lb1gt_2,456). happensAt(walking(lb1gt_2),456).
orientation(lb1gt_1,176,457). coords(lb1gt_1,183,220,457).  visible(lb1gt_1,457). happensAt(walking(lb1gt_1),457). orientation(lb1gt_2,146,457). coords(lb1gt_2,187,160,457).  visible(lb1gt_2,457). happensAt(walking(lb1gt_2),457).
orientation(lb1gt_1,176,458). coords(lb1gt_1,184,221,458).  visible(lb1gt_1,458). happensAt(walking(lb1gt_1),458). orientation(lb1gt_2,146,458). coords(lb1gt_2,187,161,458).  visible(lb1gt_2,458). happensAt(walking(lb1gt_2),458).
orientation(lb1gt_1,176,459). coords(lb1gt_1,185,222,459).  visible(lb1gt_1,459). happensAt(walking(lb1gt_1),459). orientation(lb1gt_2,146,459). coords(lb1gt_2,188,161,459).  visible(lb1gt_2,459). happensAt(walking(lb1gt_2),459).
orientation(lb1gt_1,176,460). coords(lb1gt_1,186,222,460).  visible(lb1gt_1,460). happensAt(walking(lb1gt_1),460). orientation(lb1gt_2,146,460). coords(lb1gt_2,188,162,460).  visible(lb1gt_2,460). happensAt(walking(lb1gt_2),460).
orientation(lb1gt_1,176,461). coords(lb1gt_1,187,223,461).  visible(lb1gt_1,461). happensAt(walking(lb1gt_1),461). orientation(lb1gt_2,146,461). coords(lb1gt_2,188,163,461).  visible(lb1gt_2,461). happensAt(walking(lb1gt_2),461).
orientation(lb1gt_1,176,462). coords(lb1gt_1,187,223,462).  visible(lb1gt_1,462). happensAt(walking(lb1gt_1),462). orientation(lb1gt_2,146,462). coords(lb1gt_2,188,164,462).  visible(lb1gt_2,462). happensAt(walking(lb1gt_2),462).
orientation(lb1gt_1,176,463). coords(lb1gt_1,188,224,463).  visible(lb1gt_1,463). happensAt(walking(lb1gt_1),463). orientation(lb1gt_2,146,463). coords(lb1gt_2,188,165,463).  visible(lb1gt_2,463). happensAt(walking(lb1gt_2),463).
orientation(lb1gt_1,2,464). coords(lb1gt_1,188,225,464).  visible(lb1gt_1,464). happensAt(walking(lb1gt_1),464). orientation(lb1gt_2,146,464). coords(lb1gt_2,188,168,464).  visible(lb1gt_2,464). happensAt(walking(lb1gt_2),464).
orientation(lb1gt_1,12,465). coords(lb1gt_1,188,226,465).  visible(lb1gt_1,465). happensAt(walking(lb1gt_1),465). orientation(lb1gt_2,146,465). coords(lb1gt_2,188,171,465).  visible(lb1gt_2,465). happensAt(walking(lb1gt_2),465).
orientation(lb1gt_1,12,466). coords(lb1gt_1,189,230,466).  visible(lb1gt_1,466). happensAt(walking(lb1gt_1),466). orientation(lb1gt_2,146,466). coords(lb1gt_2,188,173,466).  visible(lb1gt_2,466). happensAt(walking(lb1gt_2),466).
orientation(lb1gt_1,12,467). coords(lb1gt_1,189,232,467).  visible(lb1gt_1,467). happensAt(walking(lb1gt_1),467). orientation(lb1gt_2,146,467). coords(lb1gt_2,188,175,467).  visible(lb1gt_2,467). happensAt(walking(lb1gt_2),467).
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
#show holdsAt/2.