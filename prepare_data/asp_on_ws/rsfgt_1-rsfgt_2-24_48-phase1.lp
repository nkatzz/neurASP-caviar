
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

	orientation(rsfgt_1,170,24). coords(rsfgt_1,192,167,24).  visible(rsfgt_1,24). happensAt(walking(rsfgt_1),24). orientation(rsfgt_2,170,24). coords(rsfgt_2,180,217,24).  visible(rsfgt_2,24). happensAt(walking(rsfgt_2),24).
orientation(rsfgt_1,170,25). coords(rsfgt_1,191,166,25).  visible(rsfgt_1,25). happensAt(walking(rsfgt_1),25). orientation(rsfgt_2,170,25). coords(rsfgt_2,180,220,25).  visible(rsfgt_2,25). happensAt(walking(rsfgt_2),25).
orientation(rsfgt_1,170,26). coords(rsfgt_1,191,165,26).  visible(rsfgt_1,26). happensAt(walking(rsfgt_1),26). orientation(rsfgt_2,170,26). coords(rsfgt_2,183,221,26).  visible(rsfgt_2,26). happensAt(walking(rsfgt_2),26).
orientation(rsfgt_1,170,27). coords(rsfgt_1,189,164,27).  visible(rsfgt_1,27). happensAt(walking(rsfgt_1),27). orientation(rsfgt_2,170,27). coords(rsfgt_2,186,221,27).  visible(rsfgt_2,27). happensAt(walking(rsfgt_2),27).
orientation(rsfgt_1,170,28). coords(rsfgt_1,189,163,28).  visible(rsfgt_1,28). happensAt(walking(rsfgt_1),28). orientation(rsfgt_2,170,28). coords(rsfgt_2,188,222,28).  visible(rsfgt_2,28). happensAt(walking(rsfgt_2),28).
orientation(rsfgt_1,170,29). coords(rsfgt_1,189,162,29).  visible(rsfgt_1,29). happensAt(walking(rsfgt_1),29). orientation(rsfgt_2,170,29). coords(rsfgt_2,189,222,29).  visible(rsfgt_2,29). happensAt(walking(rsfgt_2),29).
orientation(rsfgt_1,170,30). coords(rsfgt_1,189,161,30).  visible(rsfgt_1,30). happensAt(walking(rsfgt_1),30). orientation(rsfgt_2,170,30). coords(rsfgt_2,189,222,30).  visible(rsfgt_2,30). happensAt(walking(rsfgt_2),30).
orientation(rsfgt_1,150,31). coords(rsfgt_1,189,160,31).  visible(rsfgt_1,31). happensAt(walking(rsfgt_1),31). orientation(rsfgt_2,170,31). coords(rsfgt_2,190,222,31).  visible(rsfgt_2,31). happensAt(walking(rsfgt_2),31).
orientation(rsfgt_1,150,32). coords(rsfgt_1,189,159,32).  visible(rsfgt_1,32). happensAt(walking(rsfgt_1),32). orientation(rsfgt_2,170,32). coords(rsfgt_2,190,222,32).  visible(rsfgt_2,32). happensAt(walking(rsfgt_2),32).
orientation(rsfgt_1,150,33). coords(rsfgt_1,188,158,33).  visible(rsfgt_1,33). happensAt(walking(rsfgt_1),33). orientation(rsfgt_2,170,33). coords(rsfgt_2,191,223,33).  visible(rsfgt_2,33). happensAt(walking(rsfgt_2),33).
orientation(rsfgt_1,150,34). coords(rsfgt_1,188,157,34).  visible(rsfgt_1,34). happensAt(walking(rsfgt_1),34). orientation(rsfgt_2,170,34). coords(rsfgt_2,193,223,34).  visible(rsfgt_2,34). happensAt(walking(rsfgt_2),34).
orientation(rsfgt_1,150,35). coords(rsfgt_1,188,155,35).  visible(rsfgt_1,35). happensAt(walking(rsfgt_1),35). orientation(rsfgt_2,170,35). coords(rsfgt_2,193,224,35).  visible(rsfgt_2,35). happensAt(walking(rsfgt_2),35).
orientation(rsfgt_1,150,36). coords(rsfgt_1,185,152,36).  visible(rsfgt_1,36). happensAt(walking(rsfgt_1),36). orientation(rsfgt_2,170,36). coords(rsfgt_2,198,226,36).  visible(rsfgt_2,36). happensAt(walking(rsfgt_2),36).
orientation(rsfgt_1,150,37). coords(rsfgt_1,185,149,37).  visible(rsfgt_1,37). happensAt(walking(rsfgt_1),37). orientation(rsfgt_2,170,37). coords(rsfgt_2,200,228,37).  visible(rsfgt_2,37). happensAt(walking(rsfgt_2),37).
orientation(rsfgt_1,150,38). coords(rsfgt_1,185,146,38).  visible(rsfgt_1,38). happensAt(walking(rsfgt_1),38). orientation(rsfgt_2,170,38). coords(rsfgt_2,202,230,38).  visible(rsfgt_2,38). happensAt(walking(rsfgt_2),38).
orientation(rsfgt_1,150,39). coords(rsfgt_1,185,144,39).  visible(rsfgt_1,39). happensAt(walking(rsfgt_1),39). orientation(rsfgt_2,170,39). coords(rsfgt_2,203,231,39).  visible(rsfgt_2,39). happensAt(walking(rsfgt_2),39).
orientation(rsfgt_1,150,40). coords(rsfgt_1,185,141,40).  visible(rsfgt_1,40). happensAt(walking(rsfgt_1),40). orientation(rsfgt_2,170,40). coords(rsfgt_2,205,231,40).  visible(rsfgt_2,40). happensAt(walking(rsfgt_2),40).
orientation(rsfgt_1,150,41). coords(rsfgt_1,185,141,41).  visible(rsfgt_1,41). happensAt(walking(rsfgt_1),41). orientation(rsfgt_2,170,41). coords(rsfgt_2,206,233,41).  visible(rsfgt_2,41). happensAt(walking(rsfgt_2),41).
orientation(rsfgt_1,150,42). coords(rsfgt_1,185,139,42).  visible(rsfgt_1,42). happensAt(walking(rsfgt_1),42). orientation(rsfgt_2,170,42). coords(rsfgt_2,208,233,42).  visible(rsfgt_2,42). happensAt(walking(rsfgt_2),42).
orientation(rsfgt_1,150,43). coords(rsfgt_1,184,138,43).  visible(rsfgt_1,43). happensAt(walking(rsfgt_1),43). orientation(rsfgt_2,170,43). coords(rsfgt_2,208,236,43).  visible(rsfgt_2,43). happensAt(walking(rsfgt_2),43).
orientation(rsfgt_1,150,44). coords(rsfgt_1,184,137,44).  visible(rsfgt_1,44). happensAt(walking(rsfgt_1),44). orientation(rsfgt_2,170,44). coords(rsfgt_2,209,238,44).  visible(rsfgt_2,44). happensAt(walking(rsfgt_2),44).
orientation(rsfgt_1,150,45). coords(rsfgt_1,184,137,45).  visible(rsfgt_1,45). happensAt(walking(rsfgt_1),45). orientation(rsfgt_2,170,45). coords(rsfgt_2,211,240,45).  visible(rsfgt_2,45). happensAt(walking(rsfgt_2),45).
orientation(rsfgt_1,150,46). coords(rsfgt_1,184,136,46).  visible(rsfgt_1,46). happensAt(walking(rsfgt_1),46). orientation(rsfgt_2,170,46). coords(rsfgt_2,211,243,46).  visible(rsfgt_2,46). happensAt(walking(rsfgt_2),46).
orientation(rsfgt_1,150,47). coords(rsfgt_1,184,136,47).  visible(rsfgt_1,47). happensAt(walking(rsfgt_1),47). orientation(rsfgt_2,170,47). coords(rsfgt_2,211,245,47).  visible(rsfgt_2,47). happensAt(walking(rsfgt_2),47).
orientation(rsfgt_1,150,48). coords(rsfgt_1,183,134,48).  visible(rsfgt_1,48). happensAt(walking(rsfgt_1),48). orientation(rsfgt_2,170,48). coords(rsfgt_2,212,247,48).  visible(rsfgt_2,48). happensAt(walking(rsfgt_2),48).
#show holdsAt/2.