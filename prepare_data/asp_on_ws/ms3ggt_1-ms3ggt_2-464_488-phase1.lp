
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

	orientation(ms3ggt_1,170,464). coords(ms3ggt_1,226,198,464).  visible(ms3ggt_1,464). happensAt(walking(ms3ggt_1),464). orientation(ms3ggt_2,151,464). coords(ms3ggt_2,102,152,464).  visible(ms3ggt_2,464). happensAt(walking(ms3ggt_2),464).
orientation(ms3ggt_1,170,465). coords(ms3ggt_1,228,199,465).  visible(ms3ggt_1,465). happensAt(walking(ms3ggt_1),465). orientation(ms3ggt_2,151,465). coords(ms3ggt_2,101,152,465).  visible(ms3ggt_2,465). happensAt(walking(ms3ggt_2),465).
orientation(ms3ggt_1,174,466). coords(ms3ggt_1,229,201,466).  visible(ms3ggt_1,466). happensAt(walking(ms3ggt_1),466). orientation(ms3ggt_2,151,466). coords(ms3ggt_2,101,150,466).  visible(ms3ggt_2,466). happensAt(walking(ms3ggt_2),466).
orientation(ms3ggt_1,174,467). coords(ms3ggt_1,229,202,467).  visible(ms3ggt_1,467). happensAt(walking(ms3ggt_1),467). orientation(ms3ggt_2,151,467). coords(ms3ggt_2,100,148,467).  visible(ms3ggt_2,467). happensAt(walking(ms3ggt_2),467).
orientation(ms3ggt_1,174,468). coords(ms3ggt_1,229,204,468).  visible(ms3ggt_1,468). happensAt(walking(ms3ggt_1),468). orientation(ms3ggt_2,145,468). coords(ms3ggt_2,99,147,468).  visible(ms3ggt_2,468). happensAt(walking(ms3ggt_2),468).
orientation(ms3ggt_1,174,469). coords(ms3ggt_1,230,206,469).  visible(ms3ggt_1,469). happensAt(walking(ms3ggt_1),469). orientation(ms3ggt_2,145,469). coords(ms3ggt_2,99,145,469).  visible(ms3ggt_2,469). happensAt(walking(ms3ggt_2),469).
orientation(ms3ggt_1,174,470). coords(ms3ggt_1,232,208,470).  visible(ms3ggt_1,470). happensAt(walking(ms3ggt_1),470). orientation(ms3ggt_2,145,470). coords(ms3ggt_2,99,145,470).  visible(ms3ggt_2,470). happensAt(walking(ms3ggt_2),470).
orientation(ms3ggt_1,174,471). coords(ms3ggt_1,235,211,471).  visible(ms3ggt_1,471). happensAt(walking(ms3ggt_1),471). orientation(ms3ggt_2,145,471). coords(ms3ggt_2,99,145,471).  visible(ms3ggt_2,471). happensAt(walking(ms3ggt_2),471).
orientation(ms3ggt_1,174,472). coords(ms3ggt_1,237,213,472).  visible(ms3ggt_1,472). happensAt(walking(ms3ggt_1),472). orientation(ms3ggt_2,145,472). coords(ms3ggt_2,99,143,472).  visible(ms3ggt_2,472). happensAt(walking(ms3ggt_2),472).
orientation(ms3ggt_1,177,473). coords(ms3ggt_1,240,216,473).  visible(ms3ggt_1,473). happensAt(walking(ms3ggt_1),473). orientation(ms3ggt_2,158,473). coords(ms3ggt_2,99,142,473).  visible(ms3ggt_2,473). happensAt(walking(ms3ggt_2),473).
orientation(ms3ggt_1,177,474). coords(ms3ggt_1,243,218,474).  visible(ms3ggt_1,474). happensAt(walking(ms3ggt_1),474). orientation(ms3ggt_2,158,474). coords(ms3ggt_2,99,142,474).  visible(ms3ggt_2,474). happensAt(walking(ms3ggt_2),474).
orientation(ms3ggt_1,177,475). coords(ms3ggt_1,244,220,475).  visible(ms3ggt_1,475). happensAt(walking(ms3ggt_1),475). orientation(ms3ggt_2,158,475). coords(ms3ggt_2,99,141,475).  visible(ms3ggt_2,475). happensAt(walking(ms3ggt_2),475).
orientation(ms3ggt_1,177,476). coords(ms3ggt_1,246,221,476).  visible(ms3ggt_1,476). happensAt(walking(ms3ggt_1),476). orientation(ms3ggt_2,158,476). coords(ms3ggt_2,99,141,476).  visible(ms3ggt_2,476). happensAt(walking(ms3ggt_2),476).
orientation(ms3ggt_1,177,477). coords(ms3ggt_1,246,222,477).  visible(ms3ggt_1,477). happensAt(walking(ms3ggt_1),477). orientation(ms3ggt_2,158,477). coords(ms3ggt_2,98,141,477).  visible(ms3ggt_2,477). happensAt(walking(ms3ggt_2),477).
orientation(ms3ggt_1,177,478). coords(ms3ggt_1,247,223,478).  visible(ms3ggt_1,478). happensAt(walking(ms3ggt_1),478). orientation(ms3ggt_2,158,478). coords(ms3ggt_2,98,141,478).  visible(ms3ggt_2,478). happensAt(walking(ms3ggt_2),478).
orientation(ms3ggt_1,5,479). coords(ms3ggt_1,248,224,479).  visible(ms3ggt_1,479). happensAt(walking(ms3ggt_1),479). orientation(ms3ggt_2,158,479). coords(ms3ggt_2,98,139,479).  visible(ms3ggt_2,479). happensAt(walking(ms3ggt_2),479).
orientation(ms3ggt_1,14,480). coords(ms3ggt_1,248,224,480).  visible(ms3ggt_1,480). happensAt(walking(ms3ggt_1),480). orientation(ms3ggt_2,158,480). coords(ms3ggt_2,97,137,480).  visible(ms3ggt_2,480). happensAt(walking(ms3ggt_2),480).
orientation(ms3ggt_1,14,481). coords(ms3ggt_1,249,226,481).  visible(ms3ggt_1,481). happensAt(walking(ms3ggt_1),481). orientation(ms3ggt_2,158,481). coords(ms3ggt_2,97,136,481).  visible(ms3ggt_2,481). happensAt(walking(ms3ggt_2),481).
orientation(ms3ggt_1,14,482). coords(ms3ggt_1,249,228,482).  visible(ms3ggt_1,482). happensAt(walking(ms3ggt_1),482). orientation(ms3ggt_2,158,482). coords(ms3ggt_2,96,135,482).  visible(ms3ggt_2,482). happensAt(walking(ms3ggt_2),482).
orientation(ms3ggt_1,14,483). coords(ms3ggt_1,250,231,483).  visible(ms3ggt_1,483). happensAt(walking(ms3ggt_1),483). orientation(ms3ggt_2,158,483). coords(ms3ggt_2,95,135,483).  visible(ms3ggt_2,483). happensAt(walking(ms3ggt_2),483).
orientation(ms3ggt_1,14,484). coords(ms3ggt_1,250,235,484).  visible(ms3ggt_1,484). happensAt(walking(ms3ggt_1),484). orientation(ms3ggt_2,158,484). coords(ms3ggt_2,95,134,484).  visible(ms3ggt_2,484). happensAt(walking(ms3ggt_2),484).
orientation(ms3ggt_1,18,485). coords(ms3ggt_1,251,238,485).  visible(ms3ggt_1,485). happensAt(walking(ms3ggt_1),485). orientation(ms3ggt_2,158,485). coords(ms3ggt_2,95,134,485).  visible(ms3ggt_2,485). happensAt(walking(ms3ggt_2),485).
orientation(ms3ggt_1,18,486). coords(ms3ggt_1,252,240,486).  visible(ms3ggt_1,486). happensAt(walking(ms3ggt_1),486). orientation(ms3ggt_2,158,486). coords(ms3ggt_2,95,133,486).  visible(ms3ggt_2,486). happensAt(walking(ms3ggt_2),486).
orientation(ms3ggt_1,22,487). coords(ms3ggt_1,253,242,487).  visible(ms3ggt_1,487). happensAt(walking(ms3ggt_1),487). orientation(ms3ggt_2,150,487). coords(ms3ggt_2,95,133,487).  visible(ms3ggt_2,487). happensAt(walking(ms3ggt_2),487).
orientation(ms3ggt_1,22,488). coords(ms3ggt_1,255,244,488).  visible(ms3ggt_1,488). happensAt(walking(ms3ggt_1),488). orientation(ms3ggt_2,150,488). coords(ms3ggt_2,94,133,488).  visible(ms3ggt_2,488). happensAt(walking(ms3ggt_2),488).
#show holdsAt/2.