
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

	orientation(rsfgt_0,150,24). coords(rsfgt_0,115,124,24).  visible(rsfgt_0,24). happensAt(walking(rsfgt_0),24). orientation(rsfgt_2,170,24). coords(rsfgt_2,180,217,24).  visible(rsfgt_2,24). happensAt(walking(rsfgt_2),24).
orientation(rsfgt_0,150,25). coords(rsfgt_0,113,124,25).  visible(rsfgt_0,25). happensAt(walking(rsfgt_0),25). orientation(rsfgt_2,170,25). coords(rsfgt_2,180,220,25).  visible(rsfgt_2,25). happensAt(walking(rsfgt_2),25).
orientation(rsfgt_0,150,26). coords(rsfgt_0,112,124,26).  visible(rsfgt_0,26). happensAt(walking(rsfgt_0),26). orientation(rsfgt_2,170,26). coords(rsfgt_2,183,221,26).  visible(rsfgt_2,26). happensAt(walking(rsfgt_2),26).
orientation(rsfgt_0,150,27). coords(rsfgt_0,109,124,27).  visible(rsfgt_0,27). happensAt(walking(rsfgt_0),27). orientation(rsfgt_2,170,27). coords(rsfgt_2,186,221,27).  visible(rsfgt_2,27). happensAt(walking(rsfgt_2),27).
orientation(rsfgt_0,150,28). coords(rsfgt_0,108,124,28).  visible(rsfgt_0,28). happensAt(walking(rsfgt_0),28). orientation(rsfgt_2,170,28). coords(rsfgt_2,188,222,28).  visible(rsfgt_2,28). happensAt(walking(rsfgt_2),28).
orientation(rsfgt_0,150,29). coords(rsfgt_0,107,124,29).  visible(rsfgt_0,29). happensAt(walking(rsfgt_0),29). orientation(rsfgt_2,170,29). coords(rsfgt_2,189,222,29).  visible(rsfgt_2,29). happensAt(walking(rsfgt_2),29).
orientation(rsfgt_0,150,30). coords(rsfgt_0,106,125,30).  visible(rsfgt_0,30). happensAt(walking(rsfgt_0),30). orientation(rsfgt_2,170,30). coords(rsfgt_2,189,222,30).  visible(rsfgt_2,30). happensAt(walking(rsfgt_2),30).
orientation(rsfgt_0,150,31). coords(rsfgt_0,106,125,31).  visible(rsfgt_0,31). happensAt(walking(rsfgt_0),31). orientation(rsfgt_2,170,31). coords(rsfgt_2,190,222,31).  visible(rsfgt_2,31). happensAt(walking(rsfgt_2),31).
orientation(rsfgt_0,150,32). coords(rsfgt_0,105,125,32).  visible(rsfgt_0,32). happensAt(walking(rsfgt_0),32). orientation(rsfgt_2,170,32). coords(rsfgt_2,190,222,32).  visible(rsfgt_2,32). happensAt(walking(rsfgt_2),32).
orientation(rsfgt_0,150,33). coords(rsfgt_0,104,125,33).  visible(rsfgt_0,33). happensAt(walking(rsfgt_0),33). orientation(rsfgt_2,170,33). coords(rsfgt_2,191,223,33).  visible(rsfgt_2,33). happensAt(walking(rsfgt_2),33).
orientation(rsfgt_0,150,34). coords(rsfgt_0,104,127,34).  visible(rsfgt_0,34). happensAt(walking(rsfgt_0),34). orientation(rsfgt_2,170,34). coords(rsfgt_2,193,223,34).  visible(rsfgt_2,34). happensAt(walking(rsfgt_2),34).
orientation(rsfgt_0,150,35). coords(rsfgt_0,104,127,35).  visible(rsfgt_0,35). happensAt(walking(rsfgt_0),35). orientation(rsfgt_2,170,35). coords(rsfgt_2,193,224,35).  visible(rsfgt_2,35). happensAt(walking(rsfgt_2),35).
orientation(rsfgt_0,160,36). coords(rsfgt_0,103,127,36).  visible(rsfgt_0,36). happensAt(walking(rsfgt_0),36). orientation(rsfgt_2,170,36). coords(rsfgt_2,198,226,36).  visible(rsfgt_2,36). happensAt(walking(rsfgt_2),36).
orientation(rsfgt_0,160,37). coords(rsfgt_0,102,127,37).  visible(rsfgt_0,37). happensAt(walking(rsfgt_0),37). orientation(rsfgt_2,170,37). coords(rsfgt_2,200,228,37).  visible(rsfgt_2,37). happensAt(walking(rsfgt_2),37).
orientation(rsfgt_0,160,38). coords(rsfgt_0,102,127,38).  visible(rsfgt_0,38). happensAt(walking(rsfgt_0),38). orientation(rsfgt_2,170,38). coords(rsfgt_2,202,230,38).  visible(rsfgt_2,38). happensAt(walking(rsfgt_2),38).
orientation(rsfgt_0,160,39). coords(rsfgt_0,99,127,39).  visible(rsfgt_0,39). happensAt(walking(rsfgt_0),39). orientation(rsfgt_2,170,39). coords(rsfgt_2,203,231,39).  visible(rsfgt_2,39). happensAt(walking(rsfgt_2),39).
orientation(rsfgt_0,160,40). coords(rsfgt_0,96,128,40).  visible(rsfgt_0,40). happensAt(walking(rsfgt_0),40). orientation(rsfgt_2,170,40). coords(rsfgt_2,205,231,40).  visible(rsfgt_2,40). happensAt(walking(rsfgt_2),40).
orientation(rsfgt_0,160,41). coords(rsfgt_0,94,129,41).  visible(rsfgt_0,41). happensAt(walking(rsfgt_0),41). orientation(rsfgt_2,170,41). coords(rsfgt_2,206,233,41).  visible(rsfgt_2,41). happensAt(walking(rsfgt_2),41).
orientation(rsfgt_0,160,42). coords(rsfgt_0,93,131,42).  visible(rsfgt_0,42). happensAt(walking(rsfgt_0),42). orientation(rsfgt_2,170,42). coords(rsfgt_2,208,233,42).  visible(rsfgt_2,42). happensAt(walking(rsfgt_2),42).
orientation(rsfgt_0,160,43). coords(rsfgt_0,93,131,43).  visible(rsfgt_0,43). happensAt(walking(rsfgt_0),43). orientation(rsfgt_2,170,43). coords(rsfgt_2,208,236,43).  visible(rsfgt_2,43). happensAt(walking(rsfgt_2),43).
orientation(rsfgt_0,160,44). coords(rsfgt_0,92,131,44).  visible(rsfgt_0,44). happensAt(walking(rsfgt_0),44). orientation(rsfgt_2,170,44). coords(rsfgt_2,209,238,44).  visible(rsfgt_2,44). happensAt(walking(rsfgt_2),44).
orientation(rsfgt_0,160,45). coords(rsfgt_0,91,131,45).  visible(rsfgt_0,45). happensAt(walking(rsfgt_0),45). orientation(rsfgt_2,170,45). coords(rsfgt_2,211,240,45).  visible(rsfgt_2,45). happensAt(walking(rsfgt_2),45).
orientation(rsfgt_0,160,46). coords(rsfgt_0,92,132,46).  visible(rsfgt_0,46). happensAt(walking(rsfgt_0),46). orientation(rsfgt_2,170,46). coords(rsfgt_2,211,243,46).  visible(rsfgt_2,46). happensAt(walking(rsfgt_2),46).
orientation(rsfgt_0,160,47). coords(rsfgt_0,91,132,47).  visible(rsfgt_0,47). happensAt(walking(rsfgt_0),47). orientation(rsfgt_2,170,47). coords(rsfgt_2,211,245,47).  visible(rsfgt_2,47). happensAt(walking(rsfgt_2),47).
orientation(rsfgt_0,160,48). coords(rsfgt_0,90,134,48).  visible(rsfgt_0,48). happensAt(walking(rsfgt_0),48). orientation(rsfgt_2,170,48). coords(rsfgt_2,212,247,48).  visible(rsfgt_2,48). happensAt(walking(rsfgt_2),48).
#show holdsAt/2.